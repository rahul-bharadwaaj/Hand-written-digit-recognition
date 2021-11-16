#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# In[3]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()


# In[4]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print(x_train[0].shape)


# In[5]:


plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()


# In[6]:


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# In[7]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)


# In[8]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


# In[9]:


y_predict = model.predict(x_test)
y_pred = []
for i in range(0,10000):
    y_pred.append(np.argmax(y_predict[i]))
#y_predict = np.argmax(y_predict[i])
cm = confusion_matrix(y_test,y_pred)

print(accuracy_score(y_test, y_pred))


# In[10]:



model.save('mnist.h')
print("Saving the model as mnist.h")

