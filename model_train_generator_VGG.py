from util import dicom2df, BalancedDataGenerator, getXY, storex, lossCurve
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from glob import glob
from collections import Counter
import os
import numpy as np
import pandas as pd
#%%
# Loading data
rle_df = pd.read_csv('train-rle.csv')
rle_df.columns = ['ImageId', 'EncodedPixels']
train_file_list = sorted(glob('dicom-images-train/*/*/*.dcm'))
metadata_df = dicom2df(train_file_list, rle_df)
labels = np.load('Y.npy')

#%%
# split train test and val
index = np.arange(0,len(metadata_df))
index_train_val, index_test, y_train_val, y_test = train_test_split(index, labels, stratify=labels, test_size=0.1, random_state=9001)
index_train, index_val, y_train, y_val = train_test_split(index_train_val, y_train_val, stratify=y_train_val, test_size=0.1, random_state=9001)

#%%
# prepare data
storex(metadata_df.iloc[index_train], 'data\\train')
storex(metadata_df.iloc[index_val], 'data\\val')
x_test, y_test = getXY(metadata_df.iloc[index_test])

#%%
datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input,
                                                          rotation_range=20,
                                                          width_shift_range=0.1,
                                                          height_shift_range=0.1,
                                                          horizontal_flip=True,
                                                          shear_range=0.1,
                                                          zoom_range=0.1,
                                                          brightness_range=(0.9, 1.1))

train_gen = datagen.flow_from_directory(directory='data\\train',
                                        target_size=(224,224),
                                        color_mode='rgb',
                                        batch_size=32,
                                        class_mode='categorical',
                                        shuffle=True,
                                        seed=9001)

val_gen = datagen.flow_from_directory(directory='data\\val',
                                        target_size=(224,224),
                                        color_mode='rgb',
                                        batch_size=32,
                                        class_mode='categorical',
                                        shuffle=True,
                                        seed=9001)

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//train_gen.batch_size
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
x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=kr)(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
#%%
for l in model.layers: print(l.name, l.trainable)
model.summary()
#%%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='categorical_crossentropy', metrics=["accuracy", "AUC"])
#%%
counter = Counter(train_gen.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
history = model.fit_generator(generator=train_gen,
                              class_weight=class_weights,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              epochs=50,
                              verbose=True,
                              validation_data=val_gen,
                              validation_steps=STEP_SIZE_VALID,
                              validation_freq=1)
#%%
lossCurve(history)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test==1, y_pred[:,1]>0.5)
auroc = roc_auc_score(y_test==1, y_pred[:,1]>0.5)
cm = confusion_matrix(y_test==1, y_pred[:,1]>0.5)
print(acc, auroc)
print(cm)

#%%


