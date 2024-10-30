#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, matthews_corrcoef, precision_score
import seaborn as sns


# In[3]:


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Step 2: Load Data
train_dataset = train_datagen.flow_from_directory(
    r"C:\Users\srinu\OneDrive\Desktop\FOREST_FIRE_DATASET\train",
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary'
)

test_dataset = test_datagen.flow_from_directory(
    r"C:\Users\srinu\OneDrive\Desktop\FOREST_FIRE_DATASET\test",
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    shuffle=False  # Ensures predictions and true labels match
)


# In[5]:


from tensorflow import keras

model1 = keras.Sequential()

# Add the Input layer explicitly
model1.add(keras.layers.Input(shape=(150, 150, 3)))

# Convolutional layers with MaxPooling
model1.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model1.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

model1.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

# Flattening the output and adding Dense layers
model1.add(keras.layers.Flatten())
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dense(1, activation='sigmoid'))  # Binary classification

# Summary of the model
model1.summary()


# In[6]:


from tensorflow import keras

model2 = keras.Sequential()

# Add the Input layer explicitly
model2.add(keras.layers.Input(shape=(150, 150, 3)))

# Convolutional layers with MaxPooling
model2.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model2.add(keras.layers.MaxPool2D(2, 2))

model2.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(keras.layers.MaxPool2D(2, 2))

model2.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(keras.layers.MaxPool2D(2, 2))

model2.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(keras.layers.MaxPool2D(2, 2))

# Flattening the output and adding Dense layers
model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dense(512, activation='relu'))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

# Summary of the model
model2.summary()


# In[8]:


model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[13]:


r1 = model1.fit(train_dataset, epochs=5)


# In[9]:


r2 = model2.fit(train_dataset, epochs=5)


# In[20]:


print("Test dataset classes:", test_dataset.classes[:10])


# In[21]:


predictions1 = (model1.predict(test_dataset) > 0.5).astype("int32")
predictions2 = (model2.predict(test_dataset) > 0.5).astype("int32")


# In[22]:


y_true = test_dataset.classes


# In[23]:


print("Shape of y_true:", y_true.shape)
print("Shape of predictions1:", predictions1.shape)
print("Shape of predictions2:", predictions2.shape)


# In[24]:


accuracy1 = accuracy_score(y_true, predictions1)
conf_mat1 = confusion_matrix(y_true, predictions1)
precision1 = precision_score(y_true, predictions1)
recall1 = recall_score(y_true, predictions1)
f1_1 = f1_score(y_true, predictions1)
mcc1 = matthews_corrcoef(y_true, predictions1)


# In[25]:


accuracy2 = accuracy_score(y_true, predictions2)
conf_mat2 = confusion_matrix(y_true, predictions2)
precision2 = precision_score(y_true, predictions2)
recall2 = recall_score(y_true, predictions2)
f1_2 = f1_score(y_true, predictions2)
mcc2 = matthews_corrcoef(y_true, predictions2)


# In[26]:


print("Model 1 Metrics:")
print("Accuracy:", accuracy1)
print("Confusion Matrix:\n", conf_mat1)
print("Precision:", precision1)
print("Recall:", recall1)
print("F1 Score:", f1_1)
print("MCC:", mcc1)


# In[27]:


print("\nModel 2 Metrics:")
print("Accuracy:", accuracy2)
print("Confusion Matrix:\n", conf_mat2)
print("Precision:", precision2)
print("Recall:", recall2)
print("F1 Score:", f1_2)
print("MCC:", mcc2)


# In[28]:


plt.plot(r2.history['loss'], label='Model 2 Loss')
plt.plot(r2.history['accuracy'], label='Model 2 Accuracy')
plt.legend()
plt.title('Model 2 Training')
plt.show()


# In[18]:


plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat1, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
plt.title('Model 1 Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[19]:


plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat2, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
plt.title('Model 2 Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[16]:


def predictImage(filename, model):  
    img1 = image.load_img(filename, target_size=(150, 150))
    plt.imshow(img1)
    
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y, axis=0)
    X = X / 255.0  # Normalize if necessary
    
    val = model.predict(X)
    if val > 0.3:  # Lower threshold to catch more "Fire" cases
        plt.xlabel("Fire", fontsize=30)
    else:
        plt.xlabel("No Fire", fontsize=30)
    
    plt.show()


# Test Single Image Prediction (using model1)
file_path = r"C:\Users\srinu\OneDrive\Desktop\FOREST_FIRE_DATASET\test\fire\0AEEBJWVHL79.jpg"
predictImage(file_path, model1)


# In[ ]:




