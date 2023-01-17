import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, smart_resize, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

HEIGHT = 224
WIDTH = 224

LR_HEIGHT = 24
LR_WIDTH = 24

def read_image(x):

    x_reshaped = img_to_array(load_img(x, target_size=(HEIGHT, WIDTH),
                          interpolation='bicubic'))
    y = img_to_array(load_img(x, target_size=(LR_HEIGHT, LR_WIDTH),
                 interpolation='bicubic'))
    y_reshaped = smart_resize(y, size=(HEIGHT, WIDTH),
                          interpolation='bicubic')
    
    return y_reshaped/255, x_reshaped/255

def load_data(path):
    names = os.listdir(path)
    images = [os.path.join(path, name) for name in names]
    
    return images

def tf_dataset(x, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

def preprocess(x):
    def f(x):
        x = x.decode()

        lr, hr = read_image(x)

        return lr, hr

    lr, hr = tf.numpy_function(f, [x], [tf.float32, tf.float32])
    
    lr.set_shape([HEIGHT, WIDTH, 3])
    hr.set_shape([HEIGHT, WIDTH, 3])
    return lr, hr