# Adapted from .ipynb file
import numpy as np
import pandas as pd
from glob import glob
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras import layers
from keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
from util import getXY, dicom2df, flattenimg, lossCurve
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.python.client import device_lib

#%%

tf.config.experimental.list_physical_devices('GPU')

# Loading data
rle_df = pd.read_csv('train-rle.csv')
rle_df.columns = ['ImageId', 'EncodedPixels']

#%%

train_file_list = sorted(glob('dicom-images-train/*/*/*.dcm'))
metadata_df = dicom2df(train_file_list, rle_df)

#%%

# x, y = getXY(metadata_df, verbose=True)

#%%

# A smaller data set?
X, Y = getXY(metadata_df, verbose=True)
# x_test, y_test = getXY(metadata_df.iloc[5000:5500], verbose=True)
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.1, random_state=9001)

#%%
# Preprocessing input
X_processed = preprocess_input(x_train)
x_test_processed = preprocess_input(x_test)


#%%
#balanced dataset
index = np.arange(0, x_train.shape[0], 1)
rus = RandomUnderSampler(random_state=9001)
i_rus, y_rus = rus.fit_resample(np.expand_dims(index, axis=-1), y_train)
x_rus = x_train[i_rus[:,0]]
print('Balanced data set: positive cases {}; negative cases {}'.format(np.sum(y_rus==1), np.sum(y_rus==0)))
#%%

y_train = OneHotEncoder().fit_transform(y_rus.reshape(-1,1)).toarray()
y_test = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()

#%%
kr = keras.regularizers.l1_l2(0.01, 0.01)
base_model = VGG16(include_top=False, weights='imagenet',
                   input_shape=(224, 224, 3))
base_model.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=kr)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=kr)(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs, outputs)

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
cm
plt.matshow(cm)


#%%
base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=(224, 224,3))

base_model.trainable = False
#for l in base_model.layers:
#    l.trainable = False
#base_model.layers[-2].trainable = True
#base_model.layers[-3].trainable = True
#base_model.layers[-4].trainable = True


inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = tf.keras.layers.Flatten()(x)
#x = tf.keras.layers.Dense(64, activation='relu')(x)
#x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.Dense(32, activation='relu')(x)
#x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)(x)
model = keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=["accuracy", "AUC"])

history_2 = model.fit(x_rus, y_train, batch_size=32, epochs=20, shuffle=True, validation_split=0.15)

#%%

lossCurve(history_2)
y_pred = model.predict(x_rus)
score = model.evaluate(x_test, y_test, verbose=0)
# print(score[0],score[1])
cm = confusion_matrix(y_train[:,0]==1, y_pred[:,0]>0.5)
cm
plt.matshow(cm)