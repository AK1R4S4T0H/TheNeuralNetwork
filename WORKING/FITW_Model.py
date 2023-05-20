# Cifar100 model updated to use faces in the wild instead
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Load and preprocess
lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
x_data = lfw_dataset.images
y_data = lfw_dataset.target

# Filter dataset
face_indices = np.where(np.isin(y_data, [0, 1]))[0]
x_data = x_data[face_indices]
y_data = y_data[face_indices]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Model
model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(50, 37, 1)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))

# Save
model.save("FITW.h5")
print("Model saved successfully!")

# Training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
