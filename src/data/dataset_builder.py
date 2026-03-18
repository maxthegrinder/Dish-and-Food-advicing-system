import tensorflow as tf
import numpy as np
import pandas as pd

class DatasetBuilder:
    """
    Constructs `tf.data.Dataset` pipelines for efficient loading.
    Demonstrates SRP by isolating dataset creation from model training.
    """
    def __init__(self, batch_size: int = 32, image_size: tuple = (224, 224)):
        self.batch_size = batch_size
        self.image_size = image_size

    def _load_img_resnet(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        return tf.keras.applications.resnet50.preprocess_input(img)

    def _load_img_vgg(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        return tf.keras.applications.vgg16.preprocess_input(img)

    def _load_and_preprocess_image(self, image_path, label, preprocess_fn):
        img = preprocess_fn(image_path)
        return img, label

    def create_image_dataset(self, df: pd.DataFrame, labels: np.ndarray, model_type='resnet') -> tf.data.Dataset:
        """Creates a tf.data.Dataset for CNN training."""
        if model_type == 'resnet':
            preprocess_fn = self._load_img_resnet
        else:
            preprocess_fn = self._load_img_vgg

        # Wrapping preprocess function to match tf.data.map signature
        def map_func(path, label):
            return self._load_and_preprocess_image(path, label, preprocess_fn)

        dataset = tf.data.Dataset.from_tensor_slices((df['Image_Path'].values, labels))
        dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_multimodal_dataset(self, df: pd.DataFrame, text_features: np.ndarray, labels: np.ndarray) -> tf.data.Dataset:
        """Creates a zipped tf.data.Dataset for multimodal training."""
        text_dataset = tf.data.Dataset.from_tensor_slices(text_features)

        img_dataset = tf.data.Dataset.from_tensor_slices(df['Image_Path'].values)
        img_dataset = img_dataset.map(self._load_img_resnet, num_parallel_calls=tf.data.AUTOTUNE)

        label_dataset = tf.data.Dataset.from_tensor_slices(labels)

        dataset = tf.data.Dataset.zip(((text_dataset, img_dataset), label_dataset))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
