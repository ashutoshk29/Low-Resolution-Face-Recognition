import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, smart_resize, img_to_array
from tensorflow.keras.models import load_model
from preprocessing import load_data
HEIGHT = 224
WIDTH = 224

LR_HEIGHT = 24
LR_WIDTH = 24

fecnn = load_model('models\\fecnn.h5')
srfecnn = load_model('srfecnn.h5')

gallery_path = 'dataset\\kodak'
gallery_images = load_data(gallery_path)


for test_img_path in gallery_images:

    y = img_to_array(load_img(test_img_path, target_size=(LR_HEIGHT, LR_WIDTH),
                 interpolation='bicubic'))
    y_reshaped = smart_resize(y, size=(HEIGHT, WIDTH),
                          interpolation='bicubic')/255
    y_pred = srfecnn.predict(np.expand_dims(y_reshaped, axis=0))

    losses = []
    for img_path in gallery_images:
        x_reshaped = img_to_array(load_img(img_path, target_size=(HEIGHT, WIDTH),
                          interpolation='bicubic'))/255
        y_true = fecnn.predict(np.expand_dims(x_reshaped, axis=0))
        losses.append(tf.keras.losses.mean_squared_error(y_true, y_pred).numpy()[0])

    losses = np.array(losses)

    min_index = np.where(losses == np.min(losses))
    print(min_index)