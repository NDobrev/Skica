import imageio
import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def read_data_set():
	result = np.load('../DrawingsDataSet/image_set_d345_c100.npy')
	print(result.shape)
	return result

data_set = read_data_set()
print(data_set.shape)
total_count = data_set.shape[0]
split_point = math.floor(0.9 * total_count)
print(split_point)
training_set = data_set[:split_point]
test_set =  data_set[split_point:]

training_set = training_set[... ,tf.newaxis]
test_set = test_set[... ,tf.newaxis]

print(training_set.shape)
print(test_set.shape)

# Neural net



class Autoencoder(Model):
  def __init__(self, input_shape):
    super(Autoencoder, self).__init__()
    self.model = []

    dense_layer = 0
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
    rn = tf.keras.initializers.RandomNormal(stddev=0.1)
    conv_layers = [
                   (8, (4, 4), 1, leaky_relu),
                   (16, (4, 4), 2, leaky_relu),
                   (16, (4, 4), 1, leaky_relu),
                   (32, (3, 3), 2, leaky_relu),
                   (32, (3, 3), 1, leaky_relu),
                   (64, (3, 3), 2, leaky_relu),
                   (64, (3, 3), 1, leaky_relu),
                   (64, (2, 2), 1, leaky_relu),
								   (128, (2, 2), 2, leaky_relu),
									 (64, (2, 2), 1, leaky_relu),
									# (128, (3, 3), 1, leaky_relu),
                  # (128, (3, 3), 1, leaky_relu)
                  ]

    self.current_shape = layers.Input(input_shape)
#----------------ENCODER------------------#
	
    for conv in conv_layers:
        self.add_layer(layers.Conv2D(conv[0], conv[1], strides=conv[2], activation=conv[3], padding='same', kernel_initializer=rn))
        self.add_layer(layers.BatchNormalization())

    last_conv_shape = self.current_shape.shape[1:]
    print(last_conv_shape)
    dense_layer = last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2]
    self.add_layer(layers.Flatten())
    #self.add_layer(layers.Dense(dense_layer, activation='relu'))
    #self.add_layer(layers.Dense(dense_layer / 2, activation='relu'))

#--------------LATANT-SPACE---------------#

    #self.add_layer(layers.Dense(64, activation='relu', kernel_initializer="ones",                           activity_regularizer=tf.keras.regularizers.l2(0.001)))

#----------------DECODER------------------#

    #self.add_layer(layers.Dense(dense_layer / 2, activation='relu'))
    #self.add_layer(layers.Dense(dense_layer, activation='relu'))
    self.add_layer(layers.Reshape(last_conv_shape))

    for conv in conv_layers[::-1]:
      self.add_layer(layers.Conv2DTranspose(conv[0], kernel_size=conv[1], strides=conv[2], activation=conv[3], padding='same', kernel_initializer=rn))
      self.add_layer(layers.BatchNormalization())

    self.add_layer(layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same'))

    shape = input_shape
    self.input_layer = layers.Input(input_shape)
    self.out = self.call(self.input_layer)
    # Reinitial
    super(Autoencoder, self).__init__(
            inputs=self.input_layer,
            outputs=self.out)

  def add_layer(self, layer):
    self.current_shape = layer(self.current_shape)
    self.model.append(layer)

  def call(self, x, training=False):
    current = x
    for l in self.model:
        current = l(current)
    return current

def loss_f(actual, predicted):
    r = tf.math.subtract(actual, predicted)
    r = tf.math.abs(r)
    r = tf.math.pow(r, 2)
    r = tf.math.reduce_mean(r)

    return r 

class PredictionCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    plt.clf()
    self.show_set(test_set, 0, 5, epoch)
    self.show_set(training_set, 5, 5, epoch)
    plt.draw()
    plt.pause(.01)
  def show_set(self, _set, start, n, epoch):
    random_indexes = np.random.randint(_set.shape[0], size=n)
    output_images = _set[random_indexes,:]
    decoded_imgs =  self.model.call(output_images).numpy()
		
    for i in range(n):
  # display original
      ax = plt.subplot(4, n, i + 1 + 2 * start)
      plt.imshow(output_images[i])
      plt.title("original")
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

  # display reconstruction
      ax = plt.subplot(4, n, i + 1 + n + 2* start)
      plt.imshow(decoded_imgs[i])
      plt.title(f'reconstructed {epoch}')
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)


plt.figure(figsize=(20, 4))
plt.show(block=False)


autoencoder = Autoencoder(( 32, 32, 1)) 
autoencoder.compile(Adam(lr=0.001), loss=loss_f,metrics=['accuracy'])
#autoencoder.compile(optimizer='adam', loss=loss_f)

autoencoder.build((None, 32, 32, 1))
print(autoencoder.summary())
vd=(test_set, test_set)
autoencoder.fit(training_set, training_set,
                epochs=35,
				        batch_size=64,
                shuffle=True,
                validation_data=vd,
								callbacks=[PredictionCallback()])

print("done")
plt.show(block=True)