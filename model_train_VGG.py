import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG16, InceptionV3
from util import getXY, dicom2df, flattenimg, lossCurve
from tensorflow.keras.applications.vgg16 import preprocess_input

#%%
# load data
X = np.load('X.npy')
Y = np.load('Y.npy')
#%%
# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.1, random_state=9001)
#%%
# Preprocess input
X_processed = preprocess_input(x_train)
x_test_processed = preprocess_input(x_test)
#balanced dataset
index = np.arange(0, x_train.shape[0], 1)
rus = RandomUnderSampler(random_state=9001)
i_rus, y_rus = rus.fit_resample(np.expand_dims(index, axis=-1), y_train)
x_rus = x_train[i_rus[:,0]]
print('Balanced data set: positive cases {}; negative cases {}'.format(np.sum(y_rus==1), np.sum(y_rus==0)))
#%%
# Y encoding
y_train = OneHotEncoder().fit_transform(y_rus.reshape(-1,1)).toarray()
y_test = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()

#%%
# model def
kr = tf.keras.regularizers.l1_l2(0.01, 0.01)
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=kr)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=kr)(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
#%%
for l in model.layers: print(l.name, l.trainable)
model.summary()
#%%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='categorical_crossentropy', metrics=["accuracy", "AUC"])
#%%
history = model.fit(x_rus, y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
#%%
lossCurve(history)
y_pred = model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose=0)
print(score)
cm = confusion_matrix(y_test[:,0]==1, y_pred[:,0]>0.5)
print(cm)

