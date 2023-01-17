from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from srfe_preprocessing import load_data, tf_dataset, read_image
import matplotlib.pyplot as plt

# Initialising CNN
vgg16 = VGG16(weights='imagenet')

fecnn = Sequential()
for layer in vgg16.layers[:-2]:
    fecnn.add(layer)

for layer in fecnn.layers:
    fecnn.trainable = False
print(fecnn.summary())

data_path = 'D:\\IIT\\ML in Mechanics\\FaceRecognition\\dataset\\images\\'

dataset = load_data(data_path)

dataset = tf_dataset(dataset,func = fecnn, batch=32)

srfecnn = Sequential()

srcnn = load_model('models\\srcnn.h5')
for layer in srcnn.layers:
    srfecnn.add(layer)
    
vgg16 = VGG16(weights='imagenet')

for layer in vgg16.layers[:-2]:
    srfecnn.add(layer)

print(srfecnn.summary())

srfecnn.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# Fiiting our NN to train set
r = srfecnn.fit(dataset, epochs = 10)

plt.plot(r.history['loss'][1:], label='train loss')
plt.legend()
plt.show()

srfecnn.save('srfecnn.h5')

