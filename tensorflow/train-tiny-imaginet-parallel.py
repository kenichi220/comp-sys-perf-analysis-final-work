import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import os
import json
import time
import subprocess
import pathlib
import pandas as pd

def get_num_gpus():
    try:
        count_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        num_gpus = int(count_result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        print("Nenhuma GPU foi detectada. O monitoramento não será iniciado.")
        num_gpus = 0
    return num_gpus

def generate_unique_id():
    command = 'echo "`hostname`-`date +%F`-`date +%T`"'
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout.strip()

def setup_log_directory(dir_name="logs"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def take_gpu_snapshot(unique_id, log_dir):
    filename = f"gpu_snapshot_inicial_id_{unique_id}.csv"
    snapshot_filepath = os.path.join(log_dir, filename)
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,gpu_name,driver_version,utilization.gpu,memory.total,memory.used,power.draw,temperature.gpu,clocks.gr,clocks.sm,clocks.mem",
        "--format=csv",
    ]
    with open(snapshot_filepath, "w") as f:
        subprocess.run(command, stdout=f, text=True)

def start_continuous_monitoring(unique_id, log_dir, interval_ms=500):
    processes = []
    file_handles = []
    num_gpus = get_num_gpus()
    for i in range(num_gpus):
        filename = f"gpu_monitoramento_{unique_id}_gpu_{i}.csv"
        monitoring_filepath = os.path.join(log_dir, filename)
        log_file = open(monitoring_filepath, "w")
        command = [
            "nvidia-smi", f"--id={i}",
            "--query-gpu=timestamp,index,gpu_name,driver_version,utilization.gpu,memory.total,memory.used,power.draw,temperature.gpu,clocks.gr,clocks.sm,clocks.mem",
            "--format=csv", f"-lms={interval_ms}",
        ]
        process = subprocess.Popen(command, stdout=log_file, text=True, stderr=subprocess.DEVNULL)
        processes.append(process)
        file_handles.append(log_file)
    return processes, file_handles

def stop_continuous_monitoring(processes, file_handles):
    for process in processes:
        if process.poll() is None:
            process.terminate()
            process.wait()
    for f in file_handles:
        f.close()


def load_tiny_imagenet_datasets(data_path, num_classes):
    train_path = data_path / 'train'
    val_path = data_path / 'val'

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='categorical',
        image_size=(64, 64),
        interpolation='nearest',
        batch_size=None,
        shuffle=False
    )

    val_annotations_path = val_path / 'val_annotations.txt'
    val_data = pd.read_csv(val_annotations_path, sep='\t', header=None, names=['File', 'Class', 'X1', 'Y1', 'X2', 'Y2'])
    
    class_names = sorted(os.listdir(train_path))
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    
    val_images = [str(val_path / 'images' / fname) for fname in val_data['File']]
    val_labels_str = val_data['Class'].values
    val_labels_int = [class_to_idx[name] for name in val_labels_str]
    val_labels_cat = to_categorical(val_labels_int, num_classes=num_classes)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_cat))
    
    def parse_image(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        return image, label

    val_dataset = val_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    return train_dataset, val_dataset


SEED = 1
IMG_SIZE = (224, 224)
BATCH_SIZE_PER_REPLICA = 256
NUM_CLASSES = 200
NUM_TRAIN_IMAGES = 100000

unique_id = generate_unique_id()
log_directory = setup_log_directory(dir_name="logs")
tf.random.set_seed(SEED)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
tf_config_str = os.environ.get('TF_CONFIG', '{}')
tf_config = json.loads(tf_config_str)
task_info = tf_config.get('task', {})
worker_id = task_info.get('index', 0)

print(f"Worker ID: {worker_id}")

print("Carregando o dataset Tiny ImageNet de um diretório local...")


tiny_imagenet_path = pathlib.Path('/home/users/mcogulart/tiny-imagenet/tiny-imagenet-200')

train_dataset, test_dataset = load_tiny_imagenet_datasets(tiny_imagenet_path, NUM_CLASSES)

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

NUM_WORKERS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_WORKERS

print(f"Número total de workers (réplicas): {NUM_WORKERS}")
print(f"Batch size por worker: {BATCH_SIZE_PER_REPLICA}")
print(f"Batch size global (total): {GLOBAL_BATCH_SIZE}\n")

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.with_options(options)
train_dataset = train_dataset.shuffle(NUM_TRAIN_IMAGES, seed=SEED).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.with_options(options)
test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

with strategy.scope():
    input_shape = (224, 224, 3)
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    base_model.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    BASE_LEARNING_RATE = 0.01
    SCALED_LEARNING_RATE = BASE_LEARNING_RATE * NUM_WORKERS

    opt = SGD(learning_rate=SCALED_LEARNING_RATE, weight_decay=0.0001, momentum=0.9)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

take_gpu_snapshot(unique_id, log_directory)
monitor_processes, log_files = [], []
duration_seconds = 0
try:
    monitor_processes, log_files = start_continuous_monitoring(unique_id, log_directory, 500)
    print("Iniciando o treinamento distribuído...")
    
    start_train_time = time.perf_counter()
    history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)
    close_train_time = time.perf_counter()

    duration_seconds = close_train_time - start_train_time

    score = model.evaluate(test_dataset, verbose=(1 if worker_id == 0 else 0))
    if worker_id == 0:
        print("\nIniciando avaliação final no worker 0...")
        print(f"Loss (perda) no teste: {score[0]:.4f}")
        print(f"Accuracy (acurácia) no teste: {score[1]:.4f}")
finally:
    stop_continuous_monitoring(monitor_processes, log_files)

print("TIME TRAINING:", duration_seconds)
print(f"\nWorker {worker_id} concluiu.")
