import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

#print("\nVersão do TensorFlow:", tf.__version__)

SEED = 1
tf.random.set_seed(SEED)
print("SEED number :", SEED, "\n")

print("Loading data... \n")
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

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 32
train_dataset = (
    train_dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = (
    test_dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

input_shape = (224, 224, 3)
base_model = ResNet50(None, include_top=False, input_shape=input_shape)
base_model.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

score = model.evaluate(test_dataset, verbose=0)
print(f"Loss (perda) no teste: {score[0]:.4f}")
print(f"Accuracy (acurácia) no teste: {score[1]:.4f}")
