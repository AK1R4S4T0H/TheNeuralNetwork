# Hybrid of SLL and imageDetect, i wantyed to use data augmentation on the updated model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)

# Swish
def swish(x):
    return x * K.sigmoid(x)

#  LiSHT
def lisht(x):
    return x * K.tanh(x)

# Leaky ReLU
def leaky_relu(x):
    return K.relu(x, alpha=0.1)



# model
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

train_gen = datagen.flow(x_train, y_train, batch_size=32, subset='training')
val_gen = datagen.flow(x_train, y_train, batch_size=32, subset='validation')


history = model.fit(train_gen, epochs=100, validation_data=val_gen)

model.save('SLLCNN_2.0.h5')


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
