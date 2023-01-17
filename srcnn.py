from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from preprocessing import load_data, tf_dataset

data_path = 'D:\\IIT\\ML in Mechanics\\FaceRecognition\\dataset\\images\\'

dataset = load_data(data_path)

dataset = tf_dataset(dataset)

# Initialising CNN

srnet = Sequential()
srnet.add(Conv2D(96, 9, activation='relu', padding='same', input_shape=(224, 224, 3)))
srnet.add(Conv2D(64, 1, padding='same', activation='relu'))
srnet.add(Conv2D(48, 1, padding='same', activation='relu'))
srnet.add(Conv2D(32, 1, padding='same', activation='relu'))
srnet.add(Conv2D(3, 5, padding='same', activation='relu'))

srnet.summary()

srnet.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# Fiiting our NN to train set
srnet.fit(dataset, epochs = 10)

srnet.save('srcnn.h5')


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, smart_resize, img_to_array

HEIGHT = 224
WIDTH = 224

LR_HEIGHT = 24
LR_WIDTH = 24

img_path = 'dataset\\images\\110592002_1.jpg'
x_reshaped = img_to_array(load_img(img_path, target_size=(HEIGHT, WIDTH),
                          interpolation='bicubic'))/255
y = img_to_array(load_img(img_path, target_size=(LR_HEIGHT, LR_WIDTH),
                 interpolation='bicubic'))
y_reshaped = smart_resize(y, size=(HEIGHT, WIDTH),
                          interpolation='bicubic')/255

img_array = np.expand_dims(y_reshaped, axis=0)
pred = srnet.predict(img_array)

img2 = tf.keras.preprocessing.image.array_to_img(x_reshaped)
img2.show()

img1 = tf.keras.preprocessing.image.array_to_img(img_array[0])
img1.show()

img = tf.keras.preprocessing.image.array_to_img(pred[0])
img.show()