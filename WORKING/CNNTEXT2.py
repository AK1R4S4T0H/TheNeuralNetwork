# CNN for Text generation, with Tokenizer and Model saves
import tensorflow as tf
import numpy as np
import urllib.request
import zipfile
import re
import string
import pickle
import bz2
from keras.preprocessing.text import Tokenizer
from tensorflow.python.client import device_lib

print('Good day Pilot! here\'s what you\'ll be flying /n___________________________')
print(tf.test.gpu_device_name())
print(device_lib.list_local_devices())

# Download and extract the Shakespeare dataset
url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index11.txt-p5399367p6899366.bz2'
urllib.request.urlretrieve(url, 'enwiki-latest-pages-articles-multistream-index11.txt-p5399367p6899366.bz2')
with bz2.open('/content/enwiki-latest-pages-articles-multistream-index11.txt-p5399367p6899366.bz2', 'rt') as file:
    text = file.read()

print('receiving and reading done, ion cannons primed/n_____________________________')

text = re.sub(r'\[[^]]*\]', '', text)
text = text.translate(str.maketrans('', '', string.punctuation))
text = text.lower()

# tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)

print('tokenizer pickled, engines firing up/n_______________________________')

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert text to sequences of integer indices
sequences = tokenizer.texts_to_sequences(text)
data = tf.keras.preprocessing.sequence.pad_sequences(sequences)

print('text to sequence done, pulse drives ready/n_____________________________')

def data_generator(seq_length, data, batch_size):
    while True:
        X = []
        y = []
        for i in range(batch_size):
            # choose a random sequence
            idx = np.random.randint(seq_length, len(data) - 1)
            X.append(data[idx-seq_length:idx])
            y.append(data[idx])
        X = np.array(X)
        y = np.array(y)
        y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index))
        yield X, y

seq_length = 100
batch_size = 64
train_generator = data_generator(seq_length, data, batch_size)

print('input prepared, going into deep travel, see you soon pilot /n__________')
# model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(128, 5, strides=1, input_shape=(seq_length, 1), padding='causal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(256, 5, strides=1, padding='causal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(128, 5, strides=1, padding='causal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(256, 5, strides=1, padding='causal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(tokenizer.word_index), activation='softmax')
])
print('congratulations Pilot!, you made it!!/n____________________________')
# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam')
print('warning!! aliens! train on them, suckers/n___________________________')
# Train
model.fit_generator(data_generator(seq_length, data, batch_size), epochs=100, steps_per_epoch=1000, verbose=1)
print('whew, good job again pilot, safe flying out there/n___________________________')
# Save
model.save('text_generation_model.h5')
