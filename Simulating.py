import bpy
#import cv2
import os
import numpy as np
import time, math

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
    
def CameraVelocity(vel):
    #scene.camera.rotation_euler[0] += pose[3] *(pi/180.0)
    #scene.camera.rotation_euler[1] += pose[4] *(pi/180.0)
    #scene.camera.rotation_euler[2] += pose[5]  *(pi/180.0)
    
    # Set camera translation
    scene.camera.location.x += vel[0,0] 
    scene.camera.location.y += vel[0,1] 
    scene.camera.location.z += vel[0,2]


def getPose():
    cam = bpy.data.objects['Camera']
    return (cam.matrix_world)
    # or
    cam.location
    cam.rotation_euler
    
def getCamPose():
    cam = bpy.data.objects['Camera']
    
    return ([cam.location.x, cam.location.y, cam.location.z-ORIGIN, 0,0,0])
    # or
    cam.location
    cam.rotation_euler
    
   


###########################################################################

TEMP = "/home/dvs/dvs_ws/src/pose-estimation-ROS/DeepPoseEstimation/dataset/"

import tensorflow as tf
import keras
from keras.models import load_model

   
import keras.backend as K

def keras_loss(y_actual, y_predicted):
    val = K.mean( K.sum( K.square( y_actual[:3] - y_predicted[:3] ) ) + B*K.sum( K.square( y_actual[3:] - y_predicted[3:] ) ) )
    return val


model = keras.models.load_model(TEMP + "POSE_v2.h5", custom_objects={ 'keras_loss': keras_loss })

###########################################################################




def captureImage(path=TEMP, id=0):
    bpy.context.scene.render.filepath = path + str(id) + ".png"
    bpy.ops.render.render(write_still = True)


def simulate(name="simulation", num=100):
    path = TEMP+name+"/"
    try:
        os.mkdir(path)
    except OSError as error:
        pass
    
    
    count = 1
    
    n1 = int(num*0.5)
    n2 = int(num*0.25)
    
    poseComb = np.empty((0,6), int)
    
    error = lambda p : math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    
    pose = [0.5,0,0,0,0,0]
    move(pose) 
    
    
    while error(pose) > 0.1:
        #pose = np.random.rand(6)*RANGE - RANGE/2
        pose = getCamPose()
        nppose = np.array([pose])
        poseComb = np.append(poseComb, nppose, axis=0)
        
        captureImage(path, count)
        time.sleep(0.03)
        
        loc = path + str(count) + ".png"
        image = tf.keras.preprocessing.image.load_img(loc, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image) / 255.0
        Y = model.predict(np.array([image]))
        print(Y)
        
        #vel = nppose*-1*0.1
        vel = Y*-1*0.05
        
        CameraVelocity(vel)
        
        count+=1
        
    np.savetxt(path + "/pose.csv", poseComb, fmt='%1.3f', delimiter=",")
    #fixCsv(name)


simulate()
#fixCsv("test")
#GenData("train", 5000)
#GenData("test", 500)

#poseTxt = open("./pose.txt", "w")
#poseTxt.close()

############################################################################




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
        
        if count > 500:
            break
    
    Yn = np.asarray(Y, dtype=np.float32) * np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
    
    return np.asarray(X), Yn




#model.fit(traindata, epochs = 1, batch_size = 64, validation_data= testdata)


#savePath()
#bpy.ops.render.render(write_still = True)


#poseTxt.close()

