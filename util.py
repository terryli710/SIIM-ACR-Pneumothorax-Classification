# Util functions
# dicom_to_dict and visualize_img are modified from
# https://www.kaggle.com/ekhtiar/finding-pneumo-part-1-eda-and-unet/data
import os
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt

# Image related
def makeRgb(grey_img):
    return np.stack((grey_img,) * 3, axis=-1)

# Data related
def getXY(metadata_df, verbose=False):
    """
    From metadata to trainable X and Y
    """
    def getLabel(row, verbose=False):
        l = row['encoded_pixels_list']
        if len(l) == 1 and l[0] == '-1':
            return 0
        else:
            return 1

    im_width, im_height, im_ch = 256, 256, 3
    X = np.zeros((len(metadata_df), im_width, im_height , im_ch), dtype='int8')
    Y = np.zeros(len(metadata_df))

    for index, row in tqdm(metadata_df.iterrows()):
        dataset = pydicom.dcmread(row['file_path'])
        grey_img = dataset.pixel_array
        reized_img = resize(grey_img, output_shape=(256, 256))
        stacked_img = makeRgb(reized_img)
        X[index,:,:,:] = stacked_img
        Y[index] = getLabel(row, verbose)

    assert X.shape[0] == Y.shape[0], "Length differ."
    if verbose:
        print('{} images extracted of shape {}'.format(Y.shape[0], X.shape[1:]))
        print('Found {} positive cases and {} negative cases'.format(np.sum(Y==1), np.sum(Y==0)))
    return X, Y




def dicom2dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation.

    Returns:
        dict: contains metadata of relevant fields.
    """

    data = {}

    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID

    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId'] == dicom_data.SOPInstanceUID]['EncodedPixels'].values

        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True

        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)

    return data

def dicom2df(file_path_list, rle_df):
    metadata_list = []
    for file_path in tqdm(file_path_list):
        dicom_data = pydicom.dcmread(file_path)
        train_metadata = dicom2dict(dicom_data, file_path, rle_df)
        metadata_list.append(train_metadata)
    metadata_df = pd.DataFrame(metadata_list)
    return metadata_df


# Visualization related
def visualize_img(metadata_df, index=False, num_img=3):
    """
    :param metadata_df: Dataframe from dicom (containing 'file_path')
    :param index: index to view, if False, show random index
    :param num_img: number of image to display
    """
    subplot_count = 0
    fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img * 10, 10))
    if index: df_sample = metadata_df.iloc[range(index,index+num_img,1)]
    else: df_sample = metadata_df.sample(n=num_img)
    for index, row in df_sample.iterrows():
        dataset = pydicom.dcmread(row['file_path'])
        ax[subplot_count].imshow(dataset.pixel_array, cmap=plt.cm.bone)
        # label the x-ray with information about the patient
        ax[subplot_count].text(0, 0, 'Age:{}, Sex: {}, Pneumothorax: {}'.format(row['patient_age'], row['patient_sex'],
                                                                                row['has_pneumothorax']),
                               size=26, color='white', backgroundcolor='black')
        subplot_count += 1

