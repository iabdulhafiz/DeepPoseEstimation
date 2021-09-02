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


TEMP = "/tmp/memory/"

def captureImage(path=TEMP, id=0):
    bpy.context.scene.render.filepath = path + str(id) + ".png"
    bpy.ops.render.render(write_still = True)

def GenData(name="val", num=100):
    path = TEMP+name+"/"
    try:
        os.mkdir(path)
    except OSError as error:
        pass
    
    file = open(path + "/pose.csv", "w")
    txt = ""
    count = 1
    
    n1 = int(num*0.5)
    n2 = int(num*0.25)
    
    poseComb = np.empty((0,6), int)
    
    for i in range(n1):
        pose = np.random.rand(6)*RANGE - RANGE/2
        poseComb = np.append(poseComb, np.array([pose]), axis=0)
        move(pose)
        captureImage(path, count)
        count+=1
    
    for i in range(n2):
        pose = 2*np.random.rand(6)*RANGE - 2*RANGE/2
        poseComb = np.append(poseComb, np.array([pose]), axis=0)
        move(pose)
        captureImage(path, count)
        count+=1
    
    for i in range(n2):
        pose = 0.2*np.random.rand(6)*RANGE - 0.2*RANGE/2
        poseComb = np.append(poseComb, np.array([pose]), axis=0)
        move(pose)
        captureImage(path, count)
        count+=1
        
    
    
    np.savetxt(path + "/pose.csv", poseComb, delimiter=",")
    
#GenData("train", 5000)
#GenData("test", 500)

#poseTxt = open("./pose.txt", "w")
#poseTxt.close()

############################################################################


from keras.models import load_model
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint



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



img_height,img_width = 224,224 
#If imagenet weights are being loaded, 
#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
poses = Dense(6)(x)
model = Model(inputs = base_model.input, outputs = poses)


from keras.optimizers import SGD, Adam
import keras.backend as kb
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)

def keras_loss(y_actual, y_predicted):
    val = kb.mean( kb.sum( kb.square( y_actual - y_predicted ) ) )

model.compile(optimizer= adam, loss=keras_loss, metrics=[keras_loss])


model.fit(X_train, Y_train, epochs = 100, batch_size = 64)


savePath()
bpy.ops.render.render(write_still = True)









#poseTxt.close()


