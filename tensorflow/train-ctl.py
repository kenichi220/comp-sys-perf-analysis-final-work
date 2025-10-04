
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import os
import json
import time

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

# gera um nome unico para cada analise
def generate_unique_id():
    result = subprocess.run(["date", "+%s"], capture_output=True, text=True)
    return result.stdout.strip()

# simplesmente cria um diretorio para colocar os logs das gpus durante o treinamento e retorna o caminho ate o diretorio
def setup_log_directory(dir_name="logs"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

# salva as informações das gpus antes do treinamento
def take_gpu_snapshot(unique_id, log_dir):
    filename = f"gpu_snapshot_inicial_id_{unique_id}.csv"
    snapshot_filepath = os.path.join(log_dir, filename)

    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,gpu_name,driver_version,utilization.gpu,memory.total,memory.used,power.draw,temperature.gpu",
        "--format=csv",
    ]
    with open(snapshot_filepath, "w") as f:
        subprocess.run(command, stdout=f, text=True)

# roda o monitoramento em cada gpu
def start_continuous_monitoring(unique_id, log_dir, interval_ms=500):
    processes = []
    file_handles = []

    csv_header = "timestamp,gpu_index,power.draw,temperature.gpu\n"

    num_gpus = get_num_gpus()
    for i in range(num_gpus):
        filename = f"gpu_monitoramento_{unique_id}_gpu_{i}.csv"
        monitoring_filepath = os.path.join(log_dir, filename)

        log_file = open(monitoring_filepath, "w")
        log_file.write(csv_header)
        log_file.flush()

        command = [
            "nvidia-smi",
            f"--id={i}",
            "--query-gpu=timestamp,index,gpu_name,driver_version,utilization.gpu,memory.total,memory.used,power.draw,temperature.gpu",
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

unique_id = generate_unique_id()
log_directory = setup_log_directory(dir_name = "logs")
# tf.debugging.set_log_device_placement(True)

# communication_options = tf.distribute.experimental.CommunicationOptions(
#     all_reduce_alg='ring')

strategy = tf.distribute.MultiWorkerMirroredStrategy()

tf_config_str = os.environ.get('TF_CONFIG', '{}')
tf_config = json.loads(tf_config_str)
task_info = tf_config.get('task', {})
worker_id = task_info.get('index', 0)

print(f"Worker ID: {worker_id}")
if tf.test.is_gpu_available():
    print("GPU.")
    print(tf.test.gpu_device_name())
else:
    print("no.")
SEED = 1
tf.random.set_seed(SEED)
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

BATCH_SIZE_PER_REPLICA = 32
NUM_WORKERS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_WORKERS
ACCUMULATION_STEPS = 8      # Sync gradients every 8 steps
EPOCHS = 5
EFFECTIVE_BATCH_SIZE = GLOBAL_BATCH_SIZE * ACCUMULATION_STEPS

print(f"Número total de workers (réplicas): {NUM_WORKERS}")
print(f"Batch size por worker: {BATCH_SIZE_PER_REPLICA}")
print(f"Batch size global (total): {GLOBAL_BATCH_SIZE}\n")
print(f"Gradient accumulation steps: {ACCUMULATION_STEPS}")
print(f"Effective batch size (after accumulation): {EFFECTIVE_BATCH_SIZE}")


# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

DATA_DIR = 'cifar10_tfrecords'

def _parse_function(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    label = tf.io.parse_tensor(example['label'], out_type=tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    label = tf.reshape(label, [10])
    return image, label

# Create a dataset of all shard filenames
filenames = tf.data.Dataset.list_files(os.path.join(DATA_DIR, 'train-*.tfrecord'))

# Interleave reads from multiple files for maximum throughput
# This is the key to parallel I/O
dataset = filenames.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE
)


test_filenames = tf.data.Dataset.list_files(os.path.join(DATA_DIR, 'test-*.tfrecord'))

# --- Interleave reads from multiple files ---
test_dataset = test_filenames.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Build the rest of the high-performance pipeline
train_dataset = (
    dataset.shuffle(10000)
    .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(GLOBAL_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)


# --- Build the rest of the pipeline ---
# Note: We do NOT shuffle the test dataset
test_dataset = (
    test_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(GLOBAL_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

dist_test_dataset = strategy.experimental_distribute_dataset(test_dataset)

with tf.device('/GPU:0'):
    with strategy.scope():
        input_shape = (224, 224, 3)
        base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        predictions = tf.keras.layers.Dense(10, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        # --- Create Optimizer and Loss ---
        # Use a linear scaling rule for the learning rate with the large effective batch size
        BASE_LEARNING_RATE = 0.01
        SCALED_LEARNING_RATE = BASE_LEARNING_RATE * NUM_WORKERS * ACCUMULATION_STEPS
        optimizer = tf.keras.optimizers.SGD(learning_rate=SCALED_LEARNING_RATE, momentum=0.9)

        # Loss must have reduction=NONE for custom loops
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # --- Create Gradient Accumulators (Worker-Local Variables) ---
        gradient_accumulators = [
            tf.Variable(tf.zeros_like(var), trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
        for var in model.trainable_variables
    ]

        # --- Create Step Counter and Metrics ---
        accumulation_counter = tf.Variable(0, dtype=tf.int32, trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        take_gpu_snapshot(unique_id,log_directory)
        monitor_processes,log_files = [],[]

## 4. THE CUSTOM TRAINING STEP
@tf.function
def distributed_train_step(dist_inputs):

    def accumulation_step_fn(inputs):
        x, y = inputs

        # --- Forward and backward pass ---
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            per_example_loss = loss_fn(y, y_pred)
            # Scale loss by global batch size
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        # Get gradients for this batch
        gradients = tape.gradient(loss, model.trainable_variables)

        # Accumulate the gradients
        for i in range(len(gradient_accumulators)):
            gradient_accumulators[i].assign_add(gradients[i])


        # Update accumulation counter
        accumulation_counter.assign_add(1)

        # Update metrics for this batch
        train_loss.update_state(per_example_loss)
        train_accuracy.update_state(y, y_pred)
        return per_example_loss

    def apply_and_reset_step_fn():
        # --- THE FIX IS HERE: We now read the local values inside the replica context ---
        # This resolves the conflict.
        local_accumulators = [g.read_value() for g in gradient_accumulators]

        # COMPRESS local gradients
        compressed_grads = [tf.cast(g, tf.float16) for g in local_accumulators]

        # ALL-REDUCE the compressed tensors. This is the main network communication step.
        reduced_compressed_grads = tf.distribute.get_replica_context().all_reduce(
            tf.distribute.ReduceOp.SUM, compressed_grads)

        # DECOMPRESS and SCALE
        final_grads = [tf.cast(g, tf.float32) / ACCUMULATION_STEPS for g in reduced_compressed_grads]

        # APPLY gradients
        optimizer.apply_gradients(zip(final_grads, model.trainable_variables))

        # RESET accumulators to zero
        for i in range(len(gradient_accumulators)):
            gradient_accumulators[i].assign(tf.zeros_like(model.trainable_variables[i]))

        # RESET counter
        accumulation_counter.assign(0)


    # Run the forward/backward pass on all replicas
    per_replica_losses = strategy.run(accumulation_step_fn, args=(dist_inputs,))

    local_counter_value = strategy.experimental_local_results(accumulation_counter)[0]
    # --- If accumulation is complete, sync and update weights ---
    if tf.equal(local_counter_value % ACCUMULATION_STEPS, 0):
        strategy.run(apply_and_reset_step_fn)

    # We return the collected losses from all replicas
    # return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


with tf.device('/GPU:0'):
    try:
        print("Iniciando o treinamento distribuído...")
        print("Starting custom training loop...")
        monitor_processes, log_files = start_continuous_monitoring(
            unique_id, log_directory, interval_ms=500
        )
        for epoch in range(EPOCHS):
            start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            # Reset metrics at the start of each epoch
            train_accuracy.reset_state()
            train_loss.reset_state()

            progbar = tf.keras.utils.Progbar(target=len(x_train) // GLOBAL_BATCH_SIZE)

            for step, dist_inputs in enumerate(dist_train_dataset):
                # batch_loss = distributed_train_step(dist_inputs)
                # total_loss += batch_loss
                # num_batches += 1

                distributed_train_step(dist_inputs)

                progbar.update(step + 1, [
                    ('loss', train_loss.result()),
                    ('accuracy', train_accuracy.result())
                ])

                # Update and print progress
                # train_loss.update_state(batch_loss)
                # print(f"\rEpoch {epoch+1}, Batch {num_batches}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}", end='')

            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} finished in {epoch_time:.2f}s. Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}")

        print("\nTraining complete! 🎉")

    finally:
        stop_continuous_monitoring(monitor_processes,log_files)

print(f"\nWorker {worker_id} concluiu.")
