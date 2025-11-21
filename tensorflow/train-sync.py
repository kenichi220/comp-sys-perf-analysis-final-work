import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import os
import json
import time
from datetime import datetime
import subprocess
import pathlib
import pandas as pd
import csv
import argparse

parser = argparse.ArgumentParser(description='Configure the training.')
parser.add_argument('-m', '--model', choices=['ResNet50', 'EfficientNetB0'], required=True, help='Model to be trained.')
parser.add_argument('-b', '--batch', type=int, required=True, help='Batch size for train.')

args = parser.parse_args()

log_dir = "logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch='2,10'  # Profile batches 100 through 105
)

SEED = 1
IMG_SIZE = (224, 224)
EPOCHS = 5
BATCH_SIZE_PER_REPLICA = args.batch
NUM_CLASSES = 200
NUM_TRAIN_IMAGES = 100000
epoch_start_time = 0.0 #var global 

tf.random.set_seed(SEED)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
tf_config_str = os.environ.get('TF_CONFIG', '{}')
tf_config = json.loads(tf_config_str)
task_info = tf_config.get('task', {})
worker_id = task_info.get('index', 0)

NUM_WORKERS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_WORKERS

print(f"Worker ID: {worker_id}")
print(f"Número total de workers (réplicas): {NUM_WORKERS}")
print(f"Batch size por worker: {BATCH_SIZE_PER_REPLICA}")
print(f"Batch size global (total): {GLOBAL_BATCH_SIZE}\n")

def start_timer(epoca, logs):
    global epoch_start_time  
    epoch_start_time = time.time()

def end_timer(epoca, logs):
    global epoch_start_time
    tempo = time.time() - epoch_start_time

    if worker_id == 0:
        with open(csv_file, mode='a', newline='') as file:
            csv.writer(file).writerow([epoca,tempo])

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

print("Carregando o dataset Tiny ImageNet de um diretório local...")
tiny_imagenet_path = pathlib.Path('/scratch/kbrumati/tiny-imagenet-200') # Set here the path to the dataset
train_dataset, test_dataset = load_tiny_imagenet_datasets(tiny_imagenet_path, NUM_CLASSES)

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)#test_dataset.batch(GLOBAL_BATCH_SIZE).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

with strategy.scope():
    input_shape = (224, 224, 3)

    if args.model == 'ResNet50':
        base_model = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))
    elif args.model == 'EfficientNetB0':
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))


    base_model.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    BASE_LEARNING_RATE = 0.01
    SCALED_LEARNING_RATE = BASE_LEARNING_RATE * NUM_WORKERS

    opt = SGD(learning_rate=SCALED_LEARNING_RATE, weight_decay=0.0001, momentum=0.9)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

duration_seconds = 0
print("Iniciando o treinamento distribuído...")

timestamp = datetime.now().strftime("%d_%H%M%S")
csv_file = f"logs_sync/log_{args.model}_{timestamp}.csv"
if worker_id == 0:
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode='w', newline='') as file:
        csv.writer(file).writerow(['epoca', 'tempo'])
csv_callback = tf.keras.callbacks.LambdaCallback(on_epoch_begin=start_timer,on_epoch_end=end_timer)


start_train_time = time.perf_counter()
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset,  callbacks=[tensorboard_callback,csv_callback])
close_train_time = time.perf_counter()

duration_seconds = close_train_time - start_train_time

score = model.evaluate(test_dataset, verbose=(1 if worker_id == 0 else 0))
if worker_id == 0:
    print("\nIniciando avaliação final no worker 0...")
    print(f"Loss (perda) no teste: {score[0]:.4f}")
    print(f"Accuracy (acurácia) no teste: {score[1]:.4f}")

print("TIME TRAINING:", duration_seconds)
print(f"\nWorker {worker_id} concluiu.")
