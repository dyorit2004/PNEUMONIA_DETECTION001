import cv2
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

img_directory = 'D:/Data_Science/programs(practical)/ML_python/medical_image_processing/pneumonia_dataset/'
healthy = os.listdir(img_directory + 'NORMAL/')
pneumonia = os.listdir(img_directory + 'PNEUMONIA/')

dataset = []
label = []

input_size1 = 120

for img in healthy:
    if img.endswith('.jpeg'):
        image = cv2.imread(img_directory + 'NORMAL/' + img)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size1, input_size1))
        dataset.append(np.array(image))
        label.append(0)

for img in pneumonia:
    if img.endswith('.jpeg'):
        image = cv2.imread(img_directory + 'PNEUMONIA/' + img)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size1, input_size1))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(input_size1, input_size1, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=100, validation_data=(x_test, y_test), shuffle=False)

model.save('pneumonia2.h5')


