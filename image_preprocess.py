# This file preprocess image and store them to the format that can be read by imagegenerator
from util import getXY, dicom2df, rescaleImg, makeRgb, getOutdir, getLabelv2
import os
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import matplotlib.pyplot as plt
#%%
# Loading data
rle_df = pd.read_csv('train-rle.csv')
rle_df.columns = ['ImageId', 'EncodedPixels']

#%%
train_file_list = sorted(glob('dicom-images-train/*/*/*.dcm'))
metadata_df = dicom2df(train_file_list, rle_df)
labels = np.load('Y.npy')

