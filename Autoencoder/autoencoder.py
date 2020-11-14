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

def normalize_color(x):
	return min(1, x/ 255)

def rgb2gray(rgb):
    return (np.dot(rgb[...,:3], [0.298, 0.586, 0.143])/ 255).clip(0, 1)



def read_data_set():
	full_data_set = []
	load_max = 20000000000
	for im_path in glob.glob("../DrawingsDataSet/*.png"):
         if load_max == 0:
             break

         try:
            im = imageio.imread(im_path)
            grey = rgb2gray(im)
            full_data_set.append(grey)
            load_max = load_max -1
         except:
            print("Bad image: " + im_path)
	result = np.array(full_data_set)
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
    model = []

    dense_layer = 4 * 4 * 8
    latent_dim = 64

#----------------ENCODER------------------#

    l = layers.Conv2D(12, (4,4), activation='relu', padding='same', strides=2, name="con1_4x4")
    model.append(l);
    l = layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2, name="con2_3x3")
    model.append(l);
    l = layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2, name="con3_2x2")
    model.append(l);
    l = layers.Flatten()
    model.append(l)
    l = layers.Dense(dense_layer, activation='relu')
    model.append(l)

#--------------LATANT-SPACE---------------#

    l = layers.Dense(latent_dim, activation='relu')
    model.append(l)

#----------------DECODER------------------#

    l = layers.Dense(dense_layer, activation='relu')
    model.append(l)
    l = layers.Dense(dense_layer, activation='relu')
    model.append(l)
    l = layers.Reshape((4, 4, 8))
    model.append(l)
    l = layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same', name="tcon1_2x2")
    model.append(l)
    l = layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same', name="tcon2_3x3")
    model.append(l)
    l = layers.Conv2DTranspose(12, kernel_size=4, strides=2, activation='relu', padding='same', name="tcon3_4x4")
    model.append(l)
    l = layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')
    model.append(l)
    self.model = model

    shape = input_shape
    self.input_layer = layers.Input(input_shape)
    self.out = self.call(self.input_layer)
    # Reinitial
    super(Autoencoder, self).__init__(
            inputs=self.input_layer,
            outputs=self.out)


  def call(self, x, training=False):
    current = x
    for l in self.model:
        current = l(current)
    return current

def loss_f(actual, predicted):
    r = tf.math.subtract(actual, predicted)
    r = tf.math.abs(r)
    r = tf.math.pow(r, 3)
    r = tf.math.reduce_mean(r)
    return r

autoencoder = Autoencoder(( 32, 32, 1)) 


autoencoder.compile(optimizer='adam', loss=loss_f)
autoencoder.build((None, 32, 32, 1))
print(autoencoder.summary())
#exit()
vd=(test_set, test_set)
autoencoder.fit(training_set, training_set,
                epochs=40,
				batch_size=256,
                shuffle=True,
                validation_data=vd)

print("done")

random_indexes = np.random.randint(test_set.shape[0], size=10)
output_images = test_set[random_indexes,:]
print(output_images)
decoded_imgs =  autoencoder.call(output_images).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(output_images[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()