# Convolutional Neural Network, Trained and Tested
# uses Cifar00 dataset for training, implements 
#  Swish, LISHT, and leaky ReLU activation functions,
#  05-20-2023, 20:03, created by: AK1R4S4T0H

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from keras import backend as K
print('Good Afternoon Pilot!')
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)

print('This will be a simple Training Excercise')
# Swish
def swish(x):
    return x * K.sigmoid(x)

# LiSHT
def lisht(x):
    return x * K.tanh(x)

# Leaky ReLU
def leaky_relu(x):
    return K.relu(x, alpha=0.1)

print('performing checks')
model = Sequential()
model.add(Conv2D(64, (5, 5), activation=leaky_relu, padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (5, 5), activation=leaky_relu, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (5, 5), activation=swish, padding='same'))
model.add(Conv2D(128, (5, 5), activation=lisht, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(256, (5, 5), activation=swish, padding='same'))
model.add(Conv2D(256, (5, 5), activation=lisht, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(512, (5, 5), activation=swish, padding='same'))
model.add(Conv2D(512, (5, 5), activation=lisht, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(1024, (5, 5), activation=swish, padding='same'))
model.add(Conv2D(1024, (5, 5), activation=lisht, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2048, activation=swish))
model.add(Dropout(0.5))
model.add(Dense(1024, activation=swish))
model.add(Dropout(0.5))
model.add(Dense(512, activation=lisht))
model.add(Dropout(0.5))
model.add(Dense(256, activation=leaky_relu))
model.add(Dense(100, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('AI_LOGIC_BOOT COMMENCING PLEASE WAIT...')
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))

model.save("SLLCNN.h5")
print("ALL SYSTEMS GO! Safe Flying Out There!")

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()