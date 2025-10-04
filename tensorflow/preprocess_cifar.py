import tensorflow as tf
import numpy as np
import os

print("Loading original CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# --- Configuration ---
IMG_SIZE = (224, 224)
OUTPUT_DIR = 'cifar10_tfrecords'
NUM_SHARDS = 128
NUM_CLASSES = 10
PREPROC_BATCH_SIZE = 256

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions for TFRecord serialization
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))

def create_tf_example(image, label):
    feature = {
        'image': _bytes_feature(tf.io.serialize_tensor(image)),
        'label': _bytes_feature(tf.io.serialize_tensor(label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Main function to process a dataset
def process_and_save(images, labels, subset_name):
    print(f"\nProcessing {subset_name} data...")

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    def preprocess_image(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, IMG_SIZE)

        # --- THE FIX IS HERE ---
        # Replace the NumPy-based to_categorical with the TensorFlow-native tf.one_hot.
        # The label must be an integer, so we cast it first and remove the extra dimension.
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(tf.squeeze(label), depth=NUM_CLASSES)

        return image, label

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(PREPROC_BATCH_SIZE)

    print(f"Writing to {NUM_SHARDS} TFRecord shards...")
    num_examples = len(images)
    samples_per_shard = num_examples // NUM_SHARDS

    # This sharding logic can be simplified.
    # The important part is iterating through the dataset and writing.
    shard_index = 0
    writer = None
    for i, (image_batch, label_batch) in enumerate(dataset):
        for j in range(image_batch.shape[0]):
            index = i * PREPROC_BATCH_SIZE + j
            if index % samples_per_shard == 0:
                if writer:
                    writer.close()
                shard_path = os.path.join(OUTPUT_DIR, f'{subset_name}-{shard_index:03d}-of-{NUM_SHARDS}.tfrecord')
                writer = tf.io.TFRecordWriter(shard_path)
                shard_index += 1

            example = create_tf_example(image_batch[j], label_batch[j])
            writer.write(example.SerializeToString())
    if writer:
        writer.close()

# --- Run the processing for both training and test sets ---
process_and_save(x_train, y_train, 'train')
process_and_save(x_test, y_test, 'test')

print("\nPreprocessing to TFRecord complete! 🎉")
