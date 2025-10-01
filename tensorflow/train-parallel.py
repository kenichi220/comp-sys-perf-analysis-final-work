import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import os
import json

strategy = tf.distribute.MultiWorkerMirroredStrategy()

tf_config_str = os.environ.get('TF_CONFIG', '{}')
tf_config = json.loads(tf_config_str)
task_info = tf_config.get('task', {})
worker_id = task_info.get('index', 0)

print(f"Worker ID: {worker_id}")

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

BATCH_SIZE_PER_REPLICA = 64
NUM_WORKERS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_WORKERS

print(f"Número total de workers (réplicas): {NUM_WORKERS}")
print(f"Batch size por worker: {BATCH_SIZE_PER_REPLICA}")
print(f"Batch size global (total): {GLOBAL_BATCH_SIZE}\n")


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.with_options(options)

train_dataset = train_dataset.shuffle(50000).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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

print("Iniciando o treinamento distribuído...")
history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

if worker_id == 0:
    print("\nIniciando avaliação final no worker 0...")
    score = model.evaluate(test_dataset, verbose=1)
    print(f"Loss (perda) no teste: {score[0]:.4f}")
    print(f"Accuracy (acurácia) no teste: {score[1]:.4f}")

print(f"\nWorker {worker_id} concluiu.")
