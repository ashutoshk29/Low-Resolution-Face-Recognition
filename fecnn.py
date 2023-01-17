import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
# Initialising CNN

x = tf.random.normal([1,224,224,3])

vgg16 = VGG16(weights='imagenet')

fecnn = Sequential()
for layer in vgg16.layers[:-2]:
    fecnn.add(layer)

for layer in fecnn.layers:
    fecnn.trainable = False
fecnn.summary()

fecnn.save('fecnn.h5')