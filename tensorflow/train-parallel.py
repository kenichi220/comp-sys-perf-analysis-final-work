import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import os
import json
import numpy as np
import random
import subprocess
import os


# retorna o numero de gpus reconhecida pelo nvidia-smi
def get_num_gpus():
    try:
        count_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        num_gpus = int(count_result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        print("Nenhuma GPU foi detectada. O monitoramento não será iniciado.")
        num_gpus = 0
    return num_gpus


def get_host_name():
    hostname = subprocess.run("hostname", capture_output=True, text=True)
    return hostname.stdout.strip()


# gera um nome unico para cada analise
def generate_unique_id():
    command = 'echo "hostname-date +%x-date +%T.csv"'
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()


# simplesmente cria um diretorio para colocar os logs das gpus durante o treinamento e retorna o caminho ate o diretorio
def setup_log_directory(dir_name="logs"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


# salva as informações das gpus antes do treinamento
def take_gpu_snapshot(hostname, unique_id, log_dir):
    filename = f"{hostname}-{unique_id}"
    snapshot_filepath = os.path.join(log_dir, filename)

    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,gpu_name,driver_version,power.draw,temperature.gpu,memory.usage",
        "--format=csv",
    ]
    with open(snapshot_filepath, "w") as f:
        subprocess.run(command, stdout=f, text=True)


# roda o monitoramento em cada gpu
def start_continuous_monitoring(hostname, unique_id, log_dir, interval_ms=500):
    processes = []
    file_handles = []

    csv_header = "timestamp,gpu_index,power.draw,temperature.gpu\n"

    num_gpus = get_num_gpus()
    for i in range(num_gpus):
        filename = f"monitoramento-{hostname}-{unique_id}"
        monitoring_filepath = os.path.join(log_dir, filename)

        log_file = open(monitoring_filepath, "w")
        log_file.write(csv_header)
        log_file.flush()

        command = [
            "nvidia-smi",
            f"--id={i}",
            "--query-gpu=timestamp,index,power.draw,temperature.gpu,memory.usage",
            "--format=csv,noheader,nounits",
            f"-lms={interval_ms}",
        ]

        process = subprocess.Popen(
            command, stdout=log_file, text=True, stderr=subprocess.DEVNULL
        )
        processes.append(process)
        file_handles.append(log_file)

    return processes, file_handles


# encerra o monitoramento
def stop_continuous_monitoring(processes, file_handles):
    for process in processes:
        if process.poll() is None:
            process.terminate()
            process.wait()

    for f in file_handles:
        f.close()


hostname = get_host_name()
unique_id = generate_unique_id()
log_directory = setup_log_directory(dir_name="logs")
tf.debugging.set_log_device_placement(True)
# c_options=tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
# strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=c_options)
strategy = tf.distribute.MultiWorkerMirroredStrategy()

tf_config_str = os.environ.get("TF_CONFIG", "{}")
tf_config = json.loads(tf_config_str)
task_info = tf_config.get("task", {})
worker_id = task_info.get("index", 0)

print(f"Worker ID: {worker_id}")
if tf.test.is_gpu_available():
    print("GPU.")
    print(tf.test.gpu_device_name())
else:
    print("no.")
SEED = 1
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.config.experimental.enable_op_determinism()
print(f"SEED number: {SEED}\n")

print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

IMG_SIZE = (224, 224)


def resize_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


BATCH_SIZE_PER_REPLICA = 80
NUM_WORKERS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_WORKERS

print(f"Número total de workers (réplicas): {NUM_WORKERS}")
print(f"Batch size por worker: {BATCH_SIZE_PER_REPLICA}")
print(f"Batch size global (total): {GLOBAL_BATCH_SIZE}\n")


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.DATA
)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)


train_dataset = train_dataset.with_options(options)

train_dataset = (
    train_dataset.shuffle(50000, seed=SEED)
    .batch(GLOBAL_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.with_options(options)


test_dataset = (
    test_dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(GLOBAL_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

with strategy.scope():
    input_shape = (224, 224, 3)
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    base_model.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    BASE_LEARNING_RATE = 0.01
    SCALED_LEARNING_RATE = BASE_LEARNING_RATE * NUM_WORKERS

    opt = SGD(learning_rate=SCALED_LEARNING_RATE, weight_decay=0.0001, momentum=0.9)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    take_gpu_snapshot(hostname, unique_id, log_directory)
    monitor_processes, log_files = [], []
try:
    print("Iniciando o treinamento distribuído...")
    monitor_processes, log_files = start_continuous_monitoring(hostname, unique_id, 500)
    history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

    if worker_id == 0:
        print("\nIniciando avaliação final no worker 0...")
        score = model.evaluate(test_dataset, verbose=1)
        print(f"Loss (perda) no teste: {score[0]:.4f}")
        print(f"Accuracy (acurácia) no teste: {score[1]:.4f}")
finally:
    stop_continuous_monitoring(monitor_processes, log_files)

print(f"\nWorker {worker_id} concluiu.")
