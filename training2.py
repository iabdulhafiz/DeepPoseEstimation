import bpy
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

fov = 50.0

pi = 3.14159265

scene = bpy.data.scenes["Scene"]

# Set render resolution
scene.render.resolution_x = 224
scene.render.resolution_y = 224

# Set camera fov in degrees
scene.camera.data.angle = fov*(pi/180.0)
# Set camera rotation in euler angles
scene.camera.rotation_mode = 'XYZ'

def move(pose):
    scene.camera.rotation_euler[0] = pose[3] *(pi/180.0)
    scene.camera.rotation_euler[1] = pose[4] *(pi/180.0)
    scene.camera.rotation_euler[2] = pose[5]  *(pi/180.0)
    
    # Set camera translation
    scene.camera.location.x = pose[0] 
    scene.camera.location.y = pose[1] 
    scene.camera.location.z = pose[2] + ORIGIN


def getPose():
    cam = bpy.data.objects['Camera']
    return (cam.matrix_world)
    # or
    cam.location
    cam.rotation_euler
    
    

###########################################################################


TEMP = "/home/dvs/dvs_ws/src/pose-estimation-ROS/DeepPoseEstimation/dataset/"

def captureImage(path=TEMP, id=0):
    bpy.context.scene.render.filepath = path + str(id) + ".png"
    bpy.ops.render.render(write_still = True)

def fixCsv(name="val"):
    path = TEMP+name+"/pose.csv"
    lines = []
    with open(path) as f:
        lines = f.readlines()
    data = "id,x,y,z,rx,ry,rz\n"
    count = 0
    for line in lines:
        count += 1
        data += str(count) + ".png," + line
    
    f = open(path, "w")
    f.write(data)
    f.close()

  

def GenData(name="val", num=100, s=1):
    path = TEMP+name+"/"
    try:
        os.mkdir(path)
    except OSError as error:
        pass
    
    file = open(path + "/pose.csv", "w")
    txt = ""
    count = 1
    
    n1 = int(num)
    n2 = 0#int(num*0.25)
    
    poseComb = np.empty((0,6), int)
    
    for i in range(n1):
        pose = s*np.random.rand(6)*RANGE - s*RANGE/2
        poseComb = np.append(poseComb, np.array([pose]), axis=0)
        move(pose)
        captureImage(path, count)
        count+=1
    
    for i in range(n2):
        pose = 0.5*np.random.rand(6)*RANGE - 0.5*RANGE/2
        poseComb = np.append(poseComb, np.array([pose]), axis=0)
        move(pose)
        captureImage(path, count)
        count+=1
    
    for i in range(n2):
        pose = 0.05*np.random.rand(6)*RANGE - 0.05*RANGE/2
        poseComb = np.append(poseComb, np.array([pose]), axis=0)
        move(pose)
        captureImage(path, count)
        count+=1
        
    np.savetxt(path + "/pose.csv", poseComb, fmt='%1.3f', delimiter=",")
    #fixCsv(name)



#fixCsv("test")
GenData("train_2000_1", 2000, 1)
#GenData("test", 500)

#poseTxt = open("./pose.txt", "w")
#poseTxt.close()

############################################################################
'''
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
import pandas as pd

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
        
        if count == 500:
            count = 5000
        elif count == 5250:
            count == 7500
        elif count > 7750:
            break
    
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


img_height,img_width = 224,224 
#If imagenet weights are being loaded, 
#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = keras.applications.vgg16.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape= (img_height,img_width,3))

"""
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
poses = Dense(6)(x)
model = Model(inputs = base_model.input, outputs = poses)
"""

for layers in (base_model.layers)[:19]:
    print(layers)
    layers.trainable = False
    
X= base_model.layers[-2].output
X = Flatten()(X)
predictions = Dense(6)(X)
model = Model(base_model.input, predictions)

from tensorflow.keras.optimizers import Adam
import keras.backend as K
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)

def keras_loss(y_actual, y_predicted):
    val = K.mean( K.sum( K.square( y_actual[:3] - y_predicted[:3] ) ) + B*K.sum( K.square( y_actual[3:] - y_predicted[3:] ) ) )
    return val
    
#def keras_loss(y_true, y_pred):
#    return K.mean( y_true - y_pred )

model.compile(optimizer= adam, loss=keras_loss, metrics=[keras_loss])

model = keras.models.load_model(TEMP + "POSE_v3.h5", custom_objects={ 'keras_loss': keras_loss })
#model.fit(traindata, epochs = 1, batch_size = 64, validation_data= testdata)

X, Y = generateData("train10k")
model.fit(X, Y, epochs = 4, batch_size = 64)
model.save(TEMP + "POSE_v4.h5")

#savePath()
#bpy.ops.render.render(write_still = True)


#poseTxt.close()
'''



