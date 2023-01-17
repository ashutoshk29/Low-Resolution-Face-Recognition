import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, smart_resize, img_to_array
import numpy as np
import cv2

HEIGHT = 224
WIDTH = 224

LR_HEIGHT = 24
LR_WIDTH = 24

face_classifier = cv2.CascadeClassifier('models\\haarcascade_frontalface_default.xml')

def face_extractor(img):
    
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces == ():
        cropped_face = img
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]
    
    return cropped_face

def read_image(x, func):
    x_reshaped = cv2.resize(face_extractor(cv2.imread(x)), (HEIGHT, WIDTH))
    y = cv2.resize(face_extractor(cv2.imread(x)), (LR_HEIGHT, LR_WIDTH))
    y_reshaped = cv2.resize(y, (HEIGHT, WIDTH))
    x_reshaped = np.expand_dims(x_reshaped, axis=0)
    x_features = func(x_reshaped/255)
    return y_reshaped/255, np.array(x_features)

def read_image1(x, func):
    
    x_reshaped = img_to_array(load_img(x, target_size=(HEIGHT, WIDTH),
                          interpolation='bicubic'))
    y = img_to_array(load_img(x, target_size=(LR_HEIGHT, LR_WIDTH),
                 interpolation='bicubic'))
    y_reshaped = smart_resize(y, size=(HEIGHT, WIDTH),
                          interpolation='bicubic')
    x_reshaped = np.expand_dims(x_reshaped, axis=0)
    x_features = func(x_reshaped/255)
    return y_reshaped/255, np.array(x_features)


def load_data(path):
    names = os.listdir(path)
    images = [os.path.join(path, name) for name in names]
    
    return images

def tf_dataset(x, func, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(lambda x: preprocess(x, func))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

def preprocess(x, func):
    def f(x):
        x = x.decode()
        lr, features = read_image1(x, func)
        return lr, features

    lr, features = tf.numpy_function(f, [x], [tf.float32, tf.float32])
    lr.set_shape([HEIGHT, WIDTH, 3])
    features.set_shape([1, 4096])
    return lr, features
'''
data_path = 'D:\\IIT\\ML in Mechanics\\FaceRecognition\\dataset\\images\\'
a = read_image(load_data(data_path)[0], lambda x : x)
img = tf.keras.preprocessing.image.array_to_img(a[0])
img.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet')

fecnn = Sequential()
for layer in vgg16.layers[:-2]:
    fecnn.add(layer)

for layer in fecnn.layers:
    fecnn.trainable = False

x = load_data(data_path)
dataset = tf.data.Dataset.from_tensor_slices((x))
dataset = dataset.shuffle(buffer_size=5000)
dataset = dataset.map(lambda x: preprocess(x, fecnn))

for i in dataset:
    print(1)
'''