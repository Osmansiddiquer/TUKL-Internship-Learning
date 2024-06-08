import tensorflow as tf
import os;

PATH_TO_DATASET = os.path.dirname(os.path.realpath(__file__))+'\\Dataset\\'

dataset = tf.keras.preprocessing.image_dataset_from_directory(PATH_TO_DATASET)

print(dataset.class_names)
