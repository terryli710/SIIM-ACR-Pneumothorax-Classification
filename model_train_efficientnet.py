import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from efficientnet.tfkeras import EfficientNetB5, preprocess_input
from util import getXY, dicom2df, flattenimg, lossCurve
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from collections import Counter

#%%
# Load test data
x_test = np.load('x_test.npy')
y_test = np.load("y_test.npy")

#%%
# load data
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
                                        batch_size=16,
                                        class_mode='categorical',
                                        shuffle=True,
                                        seed=9001)

val_gen = datagen.flow_from_directory(directory='data\\val',
                                        target_size=(224,224),
                                        color_mode='rgb',
                                        batch_size=16,
                                        class_mode='categorical',
                                        shuffle=True,
                                        seed=9001)

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//train_gen.batch_size
#%%
# model def
kr = tf.keras.regularizers.l1_l2(0.01, 0.01)
base_model = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = tf.keras.layers.Flatten()(x)
#x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=kr)(x)
#x = tf.keras.layers.Dropout(0.2)(x)
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
                              steps_per_epoch=5,
                              epochs=5,
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

