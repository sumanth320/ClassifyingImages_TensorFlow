#!/usr/bin/env python
# coding: utf-8

# In[7]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(40, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(69, (3, 3), activation = 'relu'))

#Flattens the images for processing
model.add(layers.Flatten())
model.add(layers.Dense(77, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

#Displays the architecture of the model
model.summary()

#Compiling the Model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics =  ['accuracy'])

history = model.fit(train_images, train_labels, epochs=3, 
                    validation_data = (test_images, test_labels))

#Plotting the Accuracy graph
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc = 'upper left')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

