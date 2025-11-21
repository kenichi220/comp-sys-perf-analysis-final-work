import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import os
import json
import time
import subprocess
import pathlib
import glob
import pandas as pd
import argparse
import csv
from datetime import datetime

parser = argparse.ArgumentParser(description='Configure the training.')
parser.add_argument('-m', '--model', choices=['ResNet50', 'EfficientNetB0'], required=True, help='Model to be trained.')
parser.add_argument('-b', '--batch', type=int, required=True, help='Batch size for train.')

args = parser.parse_args()

# unique_id = generate_unique_id()
log_directory = "logs_ctl"

strategy = tf.distribute.MultiWorkerMirroredStrategy()

tf_config_str = os.environ.get('TF_CONFIG', '{}')
tf_config = json.loads(tf_config_str)
task_info = tf_config.get('task', {})
worker_id = task_info.get('index', 0)

print(f"Worker ID: {worker_id}")
if tf.test.is_gpu_available():
    print("GPU Detectada.")
    print(tf.test.gpu_device_name())
else:
    print("No GPU.")

SEED = 1
tf.random.set_seed(SEED)
print(f"SEED number: {SEED}\n")

print("Carregando o dataset Tiny ImageNet...")
tiny_imagenet_path = pathlib.Path('/scratch/kbrumati/tiny-imagenet-200')
num_classes = 200

BATCH_SIZE_PER_REPLICA = args.batch
NUM_WORKERS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_WORKERS
ACCUMULATION_STEPS = 4
EPOCHS = 5
EFFECTIVE_BATCH_SIZE = GLOBAL_BATCH_SIZE * ACCUMULATION_STEPS
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

    return train_dataset, val_dataset, len(train_dataset)


train_dataset_raw, test_dataset_raw, num_train_images = load_tiny_imagenet_datasets(tiny_imagenet_path, num_classes)

print(f"Número total de workers (réplicas): {NUM_WORKERS}")
print(f"Batch size global (total): {GLOBAL_BATCH_SIZE}\n")
print(f"Gradient accumulation steps: {ACCUMULATION_STEPS}")
print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")

# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (224, 224))
    return image, label

# train_dataset = train_dataset_raw.with_options(options)

train_dataset = train_dataset_raw.batch(GLOBAL_BATCH_SIZE).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)


# train_dataset = (
#     train_dataset.shuffle(1024, seed=SEED)
#     .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
#     .prefetch(tf.data.AUTOTUNE)
# )

# test_dataset = test_dataset_raw.with_options(options)
test_dataset =  test_dataset_raw.batch(GLOBAL_BATCH_SIZE).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)#  (
#     test_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
#     .prefetch(tf.data.AUTOTUNE)
# )

dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
dist_test_dataset = strategy.experimental_distribute_dataset(test_dataset)

with tf.device('/GPU:0'):
    with strategy.scope():
        input_shape = (224, 224, 3)

        if args.model == 'ResNet50':
            print("Modelo selecionado: ResNet50")
            base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
        elif args.model == 'EfficientNetB0':
            print("Modelo selecionado: EfficientNetB0")
            base_model = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        BASE_LEARNING_RATE = 0.01
        SCALED_LEARNING_RATE = BASE_LEARNING_RATE * NUM_WORKERS * ACCUMULATION_STEPS
        optimizer = tf.keras.optimizers.SGD(learning_rate=SCALED_LEARNING_RATE, momentum=0.9)

        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        gradient_accumulators = [
            tf.Variable(tf.zeros_like(var), trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
            for var in model.trainable_variables
        ]

        accumulation_counter = tf.Variable(0, dtype=tf.int32, trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        #take_gpu_snapshot(unique_id, log_directory)
        #monitor_processes, log_files = [], []

@tf.function
def distributed_train_step(dist_inputs):
    def accumulation_step_fn(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            per_example_loss = loss_fn(y, y_pred)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        gradients = tape.gradient(loss, model.trainable_variables)

        for i in range(len(gradient_accumulators)):
            if gradients[i] is not None:
                gradient_accumulators[i].assign_add(gradients[i])

        accumulation_counter.assign_add(1)
        train_loss.update_state(per_example_loss)
        train_accuracy.update_state(y, y_pred)
        return per_example_loss

    def apply_and_reset_step_fn():
        local_accumulators = [g.read_value() for g in gradient_accumulators]
        compressed_grads = [tf.cast(g, tf.float16) for g in local_accumulators]

        reduced_compressed_grads = tf.distribute.get_replica_context().all_reduce(
            tf.distribute.ReduceOp.SUM, compressed_grads)

        final_grads = [tf.cast(g, tf.float32) / ACCUMULATION_STEPS for g in reduced_compressed_grads]
        optimizer.apply_gradients(zip(final_grads, model.trainable_variables))

        for i in range(len(gradient_accumulators)):
            gradient_accumulators[i].assign(tf.zeros_like(model.trainable_variables[i]))
        accumulation_counter.assign(0)

    strategy.run(accumulation_step_fn, args=(dist_inputs,))

    local_counter_value = strategy.experimental_local_results(accumulation_counter)[0]
    if tf.equal(local_counter_value % ACCUMULATION_STEPS, 0):
        strategy.run(apply_and_reset_step_fn)

with tf.device('/GPU:0'):
    try:
        print("Iniciando o treinamento distribuído...")
        print("Starting custom training loop...")

        steps_per_epoch = num_train_images // GLOBAL_BATCH_SIZE

        timestamp = datetime.now().strftime("%d_%H%M%S")

        csv_file = os.path.join(log_directory,f"log_{args.model}_{timestamp}.csv")
        if worker_id == 0:
            os.makedirs(log_directory,exist_ok=True)
            with open(csv_file,mode='w' , newline ='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoca','tempo'])

        for epoch in range(EPOCHS):
            start_time = time.time()
            train_accuracy.reset_state()
            train_loss.reset_state()

            progbar = tf.keras.utils.Progbar(target=steps_per_epoch)

            for step, dist_inputs in enumerate(dist_train_dataset):
                distributed_train_step(dist_inputs)

                progbar.update(step + 1, [
                    ('loss', train_loss.result()),
                    ('accuracy', train_accuracy.result())
                ])

            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} finished in {epoch_time:.2f}s. Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}")
            if worker_id == 0:
                with open(csv_file,mode = 'a', newline ='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch,epoch_time])

        print("\nTraining complete! 🎉")

    finally:
        pass

print(f"\nWorker {worker_id} concluiu.")
