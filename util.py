# Util functions
# dicom_to_dict and visualize_img are modified from
# https://www.kaggle.com/ekhtiar/finding-pneumo-part-1-eda-and-unet/data
import os
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
from skimage.transform import resize
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import json
from keras.utils.data_utils import Sequence
from tensorflow.keras.models import model_from_json
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator

# Image related
def makeRgb(grey_img):
    return np.stack((grey_img,) * 3, axis=-1)

def rescaleImg(imgs):
    new_imgs = ((imgs - imgs.min()) * (255 / (imgs.max() - imgs.min()))).astype(int)
    return new_imgs

# Data related
def getOutdir(name, label, pos_dir, neg_dir):
    if label == 0: return os.path.join(neg_dir, name)
    elif label == 1: return os.path.join(pos_dir, name)

def getLabel(row, verbose=False):
    l = row['encoded_pixels_list']
    if len(l) == 1 and l[0] == '-1':
        return 0
    else:
        return 1

def getLabelv2(row, verbose=False):
    l = row['encoded_pixels_list']
    # if label is empty, return None
    if len(l) == 0: return None
    if len(l) == 1 and l[0] == '-1': return 0
    else: return 1

def getXY(metadata_df, verbose=False):
    """
    From metadata to trainable X and Y
    """
    im_width, im_height, im_ch = 224, 224, 3
    X = np.zeros((len(metadata_df), im_width, im_height , im_ch), dtype='int16')
    Y = np.zeros(len(metadata_df))

    for index in tqdm(range(metadata_df.shape[0])):
        row = metadata_df.iloc[index]
        dataset = pydicom.dcmread(row['file_path'])
        grey_img = dataset.pixel_array
        resized_img = resize(grey_img, output_shape=(im_width, im_height))
        stacked_img = rescaleImg(makeRgb(resized_img))
        X[index,:,:,:] = stacked_img
        Y[index] = getLabel(row, verbose)

    # X = rescaleImg(X)
    assert X.shape[0] == Y.shape[0], "Length differ."
    if verbose:
        print('{} images extracted of shape {}'.format(Y.shape[0], X.shape[1:]))
        print('Found {} positive cases and {} negative cases'.format(np.sum(Y==1), np.sum(Y==0)))
    return X, Y

class flattenimg():
    def __init__(self, size):
        self.size = size
    def flatten(self, img):
        flattened = img.reshape((img.shape[0], np.prod(img.shape[1:])))
        return flattened
    def reconstruct(self, flatten):
        img = flatten.reshape(tuple([flatten.shape[0]] + list(self.size)))
        return img


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
    print('Constructing DataFrame...')
    for file_path in tqdm(file_path_list):
        dicom_data = pydicom.dcmread(file_path)
        train_metadata = dicom2dict(dicom_data, file_path, rle_df)
        metadata_list.append(train_metadata)
    metadata_df = pd.DataFrame(metadata_list)
    return metadata_df

# balanced sampler by oversampling the minority class
class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""
    def __init__(self, x, y, datagen, batch_size=32):
        self.datagen = datagen
        self.batch_size = batch_size
        self._shape = x.shape
        datagen.fit(x)
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)

    def __len__(self):
        return self._shape[0] // self.batch_size

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()



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

# display 9 augmented images
def visualize_augmented(traingen):
    for x_batch, y_batch in traingen:
        for i in range(0,9):
            plt.subplot(330 + 1 + i)
            plt.imshow(minmax_scale(x_batch[i,:,:,0]), cmap=plt.get_cmap('gray'))
        plt.show()
        break

# Plot loss value
def lossCurve(history):
    t_loss = history.history['loss']
    v_loss = history.history['val_loss']
    epochs = range(1, len(t_loss) + 1)
    plt.plot(t_loss, label='Training Loss')
    plt.plot(v_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot histogram of X
def histX(X):
    plt.hist(np.mean(X, axis = tuple(range(1, X.ndim))))
    plt.show()


#%%
# Storage related

def saveModelResults(history, result_dict, name):
    # Set
    log_dir = os.path.join(os.getcwd(), 'model_log')
    file_dir = os.path.join(log_dir, name + '.json')
    with open(file_dir, 'wb') as file:
        json.dump(result_dict, file)
    print('Saved to ', file_dir)
    return

def storex(metadata_df, directory, img_size=(224,224,3), verbose=False):
    #create directory
    cur_dir = os.path.abspath('')
    file_dir = os.path.join(cur_dir, directory)
    #create two classes
    pos_dir = os.path.join(file_dir, 'pos')
    neg_dir = os.path.join(file_dir, 'neg')
    newdirs = [cur_dir,pos_dir,neg_dir]
    for dir in newdirs:
        if not os.path.exists(dir): os.makedirs(dir)
    #img size
    ch=1
    if len(img_size)==2: width, height = img_size
    elif len(img_size)==3: width, height, ch = img_size
    # several values
    npos, nneg = 0, 0
    for index in tqdm(range(metadata_df.shape[0])):
        row = metadata_df.iloc[index]
        label = getLabelv2(row)
        if label == 0: nneg += 1
        elif label == 1: npos += 1
        if label is not None:
            out_dir = getOutdir(str(index)+'.png', label, pos_dir, neg_dir)
            dataset = pydicom.dcmread(row['file_path'])
            grey_img = dataset.pixel_array
            resized_img = resize(grey_img, output_shape=(width, height))
            if ch == 3: resized_img = makeRgb(resized_img)
            plt.imsave(out_dir, resized_img)
        elif verbose: print('Index {} has no label'.format(index))
    if verbose: print('Store {} pos and {} neg to directory {}'.format(npos, nneg, file_dir))
    return

# save models
def saveTrainedModel(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + ".h5")
    print("Saved model to disk")

#load model
def loadTrainedModel(name):
    # load json and create model
    json_file = open(name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + ".h5")
    print("Loaded model from disk")


if __name__ == '__main__':
    '''    
    # for debug purpose
    from glob import glob
    rle_df = pd.read_csv('train-rle.csv')
    rle_df.columns = ['ImageId', 'EncodedPixels']
    train_file_list = sorted(glob('dicom-images-train/*/*/*.dcm'))
    metadata_df = dicom2df(train_file_list, rle_df)
    getXY(metadata_df[5000:5500], verbose=True)
    '''