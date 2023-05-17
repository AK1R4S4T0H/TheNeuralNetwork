# hashAI.py
___
It is very memory intensive, please be careful, and plan accordingly
## Hash Cracking with Neural Networks

This project demonstrates a proof of concept for a hash cracking program using a neural network. The program is designed to crack hashed passwords by training a neural network model on rockyou.txt and its corresponding hash values.
___
## Usage

1. Load the dataset: The program loads the "rockyou" dataset, which contains a list of common words. This dataset is used for training the neural network.

2. Preprocess the data and train the model: The `train_model()` function preprocesses the dataset by encoding the target labels and converting the words to numerical representations (hashes). It then splits the data into training and testing sets, normalizes the input data, defines the neural network architecture, compiles the model, and trains it on the training data.

3. Crack a hash value: The `ai_agent()` function takes a hash value as input and uses the trained model to predict the original word. If the prediction is successful, it returns the cracked word; otherwise, it indicates that the hash couldn't be cracked.
___
## Example

##### Here's an example usage of the program:


target_hash = generate_hash("password")
result = ai_agent(target_hash, model, label_encoder, scaler)
print(result)

This example attempts to crack the hash value of the word "password" using the trained model.
___
## Limitations
The performance of the hash cracking program depends on the quality and size of the dataset used for training. Using a larger and more diverse dataset may improve the cracking success rate.

The neural network architecture and hyperparameters used in this proof of concept may not be optimal. Experimenting with different architectures and hyperparameters could lead to better results.

This program is for educational and proof-of-concept purposes only. Hash cracking without proper authorization is illegal and unethical. Use this program responsibly and in compliance with applicable laws and regulations.

## License
This project is licensed under the MIT License.