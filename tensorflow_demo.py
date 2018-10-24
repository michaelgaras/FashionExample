# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import random

# Helper libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Importing the fashion data set
fashion_mnist = keras.datasets.fashion_mnist

# Saves pictures into testing and training lists and labels
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# labels are listed as a number 0 - 9, this array translatates numbers to words
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the data

# scaling pixel values from 0 - 255 to a range of 0 to 1 before feeding to
#the neural network model.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Setting up layers of convolutional neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #2d-array (of 28 by 28 pixels), to a 1d-array
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compiling model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=5)

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Running prediction on the testing set
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

###############DEMO#####################
i = random.randint(0,1000)
num_rows = 2
num_cols = 1
plt.figure(figsize=(10*num_cols, 7*num_rows))
plt.subplot(1, 2*num_cols, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2*num_cols, 2)
plot_value_array(i, predictions, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

#######################################






























































# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions, test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions, test_labels)
#
# #plt.show()

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
