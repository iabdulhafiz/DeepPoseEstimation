#import cv2
import os
import numpy as np
#import pandas as pd

import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)


SCALE = 1
ORIGIN = 2*SCALE
RANGE = np.array([1, 1, 1, 30, 30, 30])


pi = 3.14159265


    
    

###########################################################################


TEMP = "/home/dvs/dvs_ws/src/pose-estimation-ROS/DeepPoseEstimation/dataset/"

############################################################################

import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Reshape, Input, Dense, Dropout, Flatten, Conv2D, ConvLSTM2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
import pandas as pd


from tensorflow.keras.optimizers import Adam
import keras.backend as K

K.set_image_data_format('channels_last')

BATCH_SIZE = 32
B = 0.1

"""
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_name = df[y_col['name']].nunique()
        self.n_type = df[y_col['type']].nunique()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, bbox, target_size):
    
        xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        bbox_batch = batches[self.X_col['bbox']]
        
        name_batch = batches[self.y_col['name']]
        type_batch = batches[self.y_col['type']]

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, tuple([y0_batch, y1_batch])
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

"""


def generateData(name="val"):
    path = TEMP+name+"/pose.csv"
    lines = []
    with open(path) as f:
        lines = f.readlines()
    
    count = 0
    X = []
    Y = []
    for line in lines:
        count += 1
        Y.append(line.rstrip("\n").split(","))
        loc = TEMP + name + "/" + str(count) + ".png"
        image = tf.keras.preprocessing.image.load_img(loc, target_size=(224, 224))
        image_arr = keras.preprocessing.image.img_to_array(image) / 255.0
        X.append(image_arr)
        
        #if count == 500:
        #    count = 5000
        #elif count == 5250:
        #    count == 7500
        #elif count > 7750:
        #    break
    
    Yn = np.asarray(Y, dtype=np.float32) * np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
    
    return np.asarray(X), Yn

def generateDataLSTM(name="val"):
    path = TEMP+name+"/pose.csv"
    lines = []
    with open(path) as f:
        lines = f.readlines()
    
    count = -1
    X = []
    Y = []
    for line in lines:
        count += 1
        Y.append(line.rstrip("\n").split(","))
        
        L = []
        for i in range(2):
            loc = TEMP + name + "/" + str(count) + '.' + str(i) + ".png"
            image = tf.keras.preprocessing.image.load_img(loc, target_size=(224, 224))
            image_arr = keras.preprocessing.image.img_to_array(image) / 255.0
            L.append(image_arr)
            
        X.append(L)
        #if count == 500:
        #    count = 5000
        #elif count == 5250:
        #    count == 7500
        #elif count > 7750:
        #    break
    
    Yn = np.asarray(Y, dtype=np.float32) * np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
    
    return np.asarray(X), Yn

"""
traindf = pd.read_csv(TEMP + "test/pose.csv", dtype=str)
testdf = pd.read_csv(TEMP + "test/pose.csv", dtype=str)
label = ['x', 'y', 'z', 'rx', 'ry', 'rz']

trdata = ImageDataGenerator()
traindata = trdata.flow_from_dataframe(dataframe=traindf, directory=TEMP + "test", x_col = "id", y_col = label, batch_size=BATCH_SIZE, color_mode='rgb',shuffle=False, class_mode='raw',target_size=(224,224))

tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_dataframe(dataframe=testdf, directory=TEMP + "test", x_col = "id", y_col = label, batch_size=BATCH_SIZE, color_mode='rgb',shuffle=False, class_mode='raw',target_size=(224,224))
"""




def get_original():
    base_model = keras.applications.vgg16.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape= (224,224,3))

    for layers in (base_model.layers)[:10]:
        print(layers)
        layers.trainable = False
    #x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    #model = Model(inputs = base_model.input, outputs = poses)
    #poses = Dense(6)(x)
    X= base_model.layers[-2].output
    X = Flatten()(X)
    #X = Dense(512)(X)
    predictions = Dense(6)(X)
    model = Model(base_model.input, predictions)
    return model





def get_lstm():

    inputs = Input((2, 224, 224, 3))
    #lstm = ConvLSTM2D(32, (3,3), activation='relu', padding='same', return_sequences=True)(inputs)
    lstm = ConvLSTM2D(64, (3,3), activation='relu', padding='same', return_sequences=False)(inputs)

    #lstm = Conv2D(64, (3, 3))(lstm)
    #lstm = Reshape((224, 224, 64))(lstm)

    base_model = keras.applications.vgg16.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape= (224,224,3))

    for layers in (base_model.layers)[2:10]:
        #layers.trainable = False
        lstm = layers(lstm)

    for layers in (base_model.layers)[10:19]:
        #layers.trainable = False
        lstm = layers(lstm)

    lstm = Flatten()(lstm)
    predictions = Dense(6)(lstm)
    model = Model(inputs, predictions)

    model.summary()

    return model


def keras_loss(y_actual, y_predicted):
    val = K.mean( K.sum( K.square( y_actual[:3] - y_predicted[:3] ) ) + B*K.sum( K.square( y_actual[3:] - y_predicted[3:] ) ) )
    return val
    

#model = keras.models.load_model(TEMP + "POSE_v2.2.h5", custom_objects={ 'keras_loss': keras_loss })


def train_original():
    model = get_original()

    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss=keras_loss, metrics=[keras_loss])

    X, Y = generateData("train_2000_0.01")
    model.fit(X, Y, epochs = 30, batch_size = 64)
    model.save(TEMP + "POSE_v2.2.h5")


def train_lstm(ver=""):
    model = get_lstm()

    if ver != "":
        adam = Adam(lr=0.0001)
        model.compile(optimizer= adam, loss=keras_loss, metrics=[keras_loss])
    else:
        model = keras.models.load_model(TEMP + "POSE_LSTMv"+ver+".h5", custom_objects={ 'keras_loss': keras_loss })

    X, Y = generateDataLSTM("trainlstm_0.01")
    model.fit(X, Y, epochs = 30, batch_size = 16)
    model.save(TEMP + "POSE_LSTMv"+ver+".h5")


train_lstm()