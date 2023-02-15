# Multilayer perceptron Model (MLP) Example
#  Autoencoder to remove noise from images

# Import Libs
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import numpy as np

# Fetch data from dataset
(X_train, _), (X_test, _) = mnist.load_data()

# 1. Data preparation

# Normalize input data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Add noise to images
noise_factor = 0.6
X_train_noisy = X_train + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = X_train.shape)
X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0) # Clip range between 0, 1
#
X_test_noisy = X_test + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = X_test.shape)
X_test_noisy = np.clip(X_test_noisy, 0.0, 1.0)

# Double Check Shape of matrix
#print(X_train.shape)

# Convert input data to vector for MLP
X_train = X_train.reshape(len(X_train), 784)
X_test = X_test.reshape(len(X_test), 784)
#
X_train_noisy = X_train_noisy.reshape(len(X_train_noisy), 784)
X_test_noisy = X_test_noisy.reshape(len(X_test_noisy), 784)

# Double Check Shape of matrix
#print(X_train.shape)

# 2. Autoencoder Model Design

# Input layer
input_img = Input(shape = (784, ))
encoder = Dense(units = 32, activation = 'relu')(input_img)
# Output layer
decoder = Dense(units = 784, activation = 'sigmoid')(encoder)

autoencoder = Model(input_img, decoder)

# Check model properties
#autoencoder.summary()

# 3. Train the model

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# Unsupervised training: Input = Output
autoencoder.fit(X_train_noisy, X_train, epochs = 50, batch_size = 256)

# Encoder Only, for test

enc_model = Model(input_img, encoder)
#enc_model.summary()

# Generate test results
pred = autoencoder.predict(X_test_noisy)
encoded_images = enc_model.predict(X_test_noisy)

# Plot some samples

plt.figure(figsize = (40, 4))
for i in range(10):
  # Input image
  ax = plt.subplot(3, 20, i + 1)
  plt.imshow(X_test_noisy[i].reshape(28,28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # Encoded image
  ax = plt.subplot(3, 20, i + 1 + 20)
  plt.imshow(encoded_images[i].reshape(8,4))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # Decoded image (Output)
  ax = plt.subplot(3, 20, 2 * 20 + i + 1)
  plt.imshow(pred[i].reshape(28,28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()
