#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 22:42:54 2020

@author: mike
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
import matplotlib.pyplot as plt


train_data = np.load("train_data.npy")

x_data = np.zeros((210,204,204,3))
y_data = np.zeros(210)

for i in range(210):
    img = train_data[i,1:].reshape(1024,1024)
    img_resized = resize(img,(204,204))
    y_data[i] = train_data[i,0]
    x_data[i,:,:,0] = img_resized.astype(int)
    x_data[i,:,:,1] = img_resized.astype(int)
    x_data[i,:,:,2] = img_resized.astype(int)

x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)



y_train = OneHotEncoder().fit_transform(y_train.reshape(-1,1)).toarray()
y_test = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()
    



base_model = VGG16(include_top=False, weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                   input_shape=(204, 204, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(204, 204, 3))
x = base_model(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.summary()



model.compile(optimizer=tf.keras.optimizers.SGD(),loss="binary_crossentropy",metrics=["accuracy"])


model.fit(x_train, y_train, batch_size=16, epochs=5)


pred = model.predict(x_train)

score = model.evaluate(x_test, y_test, verbose=0)
print(score[0],score[1])