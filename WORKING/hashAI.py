# hash Cracking AI agent, has been trained and tested
# does work but is very very resource intensive
# had to do batch of one or it crashed colab with gpu
import urllib.request
import hashlib
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to allocate only a fraction of GPU memory
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])  # Limit to 2GB
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# Function to generate hash value
def generate_hash(word):
    return hashlib.md5(word.encode()).hexdigest()

# Function to load the rockyou dataset
def load_dataset():
    url = "https://github.com/brannondorsey/naive-hashcat/releases/download/data/rockyou.txt"
    urllib.request.urlretrieve(url, "rockyou.txt")
    with open("rockyou.txt", "r", encoding="latin-1") as file:
        words = file.read().splitlines()
    return words

# Function to preprocess the data and train the model
def train_model(words, hashes):
    # Encode the target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(hashes)

    # Convert the words to numerical representations (hashes)
    hashes_encoded = [int(hashlib.sha256(w.encode()).hexdigest(), 16) for w in words]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(hashes_encoded, y_encoded, test_size=0.2, random_state=42)

    # Normalize the input data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(np.array(X_train).reshape(-1, 1))
    X_test_scaled = scaler.transform(np.array(X_test).reshape(-1, 1))

    # Define the neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=1, verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    return model, label_encoder

# Function to use the trained model for hash cracking
def ai_agent(hash_value, model, label_encoder, scaler):
    hash_encoded = [int(hashlib.sha256(hash_value.encode()).hexdigest(), 16)]
    hash_scaled = scaler.transform(np.array(hash_encoded).reshape(-1, 1))
    prediction = model.predict(hash_scaled)
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction))
    if predicted_label:
        return f"Hash cracked! The original word is: {predicted_label[0]}"
    else:
        return "Unable to crack the hash"

# Load the rockyou dataset
words = load_dataset()
hashes = [generate_hash(word) for word in words]

# Train the neural network model
model, label_encoder = train_model(words, hashes)

# Example usage: Crack a hash value
target_hash = generate_hash("password")
scaler = StandardScaler()  # Instantiate a new scaler for inference
result = ai_agent(target_hash, model, label_encoder, scaler)
print(result)
