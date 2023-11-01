import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import loadtxt
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report





_CIFAR_IMAGE_SIZE = 32

def load_data(path, labels_number=1):
    with tf.io.gfile.GFile(path, 'rb') as f:
        data = f.read()
    offset = 0
    max_offset = len(data) - 1
    coarse_labels = []
    fine_labels = []
    images = []
    
    while offset < max_offset:
        labels = np.frombuffer(
            data, dtype=np.uint8, count=labels_number, offset=offset
            ).reshape((labels_number,))
        
        # 1 байт под названия, 1024 * 3 = 3072 байтов под изображение.
        offset += labels_number
        img = (
            np.frombuffer(data, dtype=np.uint8, count=3072, offset=offset)
            .reshape((3, _CIFAR_IMAGE_SIZE, _CIFAR_IMAGE_SIZE))
            .transpose((1, 2, 0))
            )

        offset += 3072
        coarse_labels.append(labels[0])
        fine_labels.append(labels[1])
        images.append(img)
        
    return [np.array(coarse_labels), np.array(fine_labels), np.array(images)]

def load_labels(path):
    return loadtxt(path, comments="#", delimiter=",", unpack=False, dtype='str')

def load_cifar100():
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    dataset = tf.keras.utils.get_file("cifar.tar.gz", url,untar=True, cache_dir='.', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'cifar-100-binary')
    CY_train, FY_train, X_train = load_data(os.path.join(dataset_dir, 'train.bin'), labels_number=2)
    CY_test, FY_test, X_test = load_data(os.path.join(dataset_dir, 'test.bin'), labels_number=2)
    C_label = load_labels(os.path.join(dataset_dir, 'coarse_label_names.txt'))
    F_label = load_labels(os.path.join(dataset_dir, 'fine_label_names.txt'))
    
    return X_train, CY_train, FY_train, X_test, CY_test, FY_test, C_label, F_label



X_train, CY_train, FY_train, X_test, CY_test, FY_test, C_label, F_label = load_cifar100()



# Нормализация значений rgb
X_train, X_test = X_train / 255.0, X_test / 255.0


# Первая модель

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(100, activation='softmax'))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])

model.fit(X_train, CY_train, epochs=5)

model.save('first.keras')


# Вторая модель


model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(256, activation='relu'))
model2.add(layers.Dense(128, activation='relu'))
model2.add(layers.Dense(100, activation='softmax'))

model2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])

model2.fit(X_train, FY_train, epochs=5)

model2.save('second.keras')


