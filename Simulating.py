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
    scene.camera.rotation_euler[0] += vel[0,3] *(pi/180.0)*5
    scene.camera.rotation_euler[1] += vel[0,4] *(pi/180.0)*5
    scene.camera.rotation_euler[2] += vel[0,5]  *(pi/180.0)*5
    
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
    #val = K.mean( K.sum( K.square( y_actual[:3] - y_predicted[:3] ) ) + B*K.sum( K.square( y_actual[3:] - y_predicted[3:] ) ) )
    #return val
    return None



###########################################################################




def captureImage(path=TEMP, id=0):
    bpy.context.scene.render.filepath = path + str(id) + ".png"
    bpy.ops.render.render(write_still = True)


def simulate(name="simulation", num=100):
    
    model = keras.models.load_model(TEMP + "POSE_v2.2.h5", custom_objects={ 'keras_loss': keras_loss })

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
    
    pose = [0,0,0,0,0,0]
    move(pose)
    captureImage(path, 0)
    
    pose = [-0.5, 0.2, 0.0, 4,-8,20]
    #pose = [0,0,0,0,0,0]
    move(pose) 
    
    
    while True:#error(pose) > 0.1:
        #pose = np.random.rand(6)*RANGE - RANGE/2
        pose = getCamPose()
        nppose = np.array([pose])
        poseComb = np.append(poseComb, nppose, axis=0)
        
        captureImage(path, count)
        #time.sleep(0.03)
        
        loc = path + str(count) + ".png"
        image = tf.keras.preprocessing.image.load_img(loc, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image) / 255.0
        Y = model.predict(np.array([image]))
        print(Y)
        
        #vel = nppose*-1*0.1
        vel = Y*-1*0.1
        
        CameraVelocity(vel)
        
        count+=1
        
        if count > 200:
            break
        
    np.savetxt(path + "/pose.csv", poseComb, fmt='%1.3f', delimiter=",")
    #fixCsv(name)


#simulate("sim3.0")
#fixCsv("test")
#GenData("train", 5000)
#GenData("test", 500)

#poseTxt = open("./pose.txt", "w")
#poseTxt.close()

############################################################################


def simulateLSTM(name="simulation", num=100):
    model = keras.models.load_model(TEMP + "POSE_LSTMv.h5", custom_objects={ 'keras_loss': keras_loss })

    
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
    
    pose = [0,0,0,0,0,0]
    move(pose)
    captureImage(path, 0)
    
    pose = [0.5, 0.2, 0.0, 4,0,5]
    #pose = [0,0,0,0,0,0]
    move(pose) 
    
    
    while True:#error(pose) > 0.1:
        #pose = np.random.rand(6)*RANGE - RANGE/2
        pose = getCamPose()
        nppose = np.array([pose])
        poseComb = np.append(poseComb, nppose, axis=0)
        
        captureImage(path, count)
        #time.sleep(0.03)
        
        loc = path + str(count) + ".png"
        image1 = tf.keras.preprocessing.image.load_img(loc, target_size=(224, 224))
        image1 = keras.preprocessing.image.img_to_array(image1) / 255.0
        
        image2 = image1
        if count != 1:
            loc = path + str(count-1) + ".png"
            image2 = tf.keras.preprocessing.image.load_img(loc, target_size=(224, 224))
            image2 = keras.preprocessing.image.img_to_array(image2) / 255.0

        
        Y = model.predict(np.array([[image1, image2]]))
        print(Y)
        
        #vel = nppose*-1*0.1
        vel = Y*-1*0.1
        
        CameraVelocity(vel)
        
        count+=1
        
        if count > 200:
            break
        
    np.savetxt(path + "/pose.csv", poseComb, fmt='%1.3f', delimiter=",")
    #fixCsv(name)
    
    
simulateLSTM("simlstm")