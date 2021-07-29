import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def image_to_vector(image) :

    return cv2.resize(image,(28,28)).flatten()

data = keras.datasets.fashion_mnist
names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

(train_image,train_label),(test_image,test_label) = data.load_data()
train_image = train_image / 255
test_image = test_image / 255

train_image_flat = []
test_image_flat = []

for i in range(len(train_image)) :

    train_image_flat.append(image_to_vector(train_image[i]))

for i in range(len(test_image)) :

    test_image_flat.append(image_to_vector(test_image[i]))

# KNN MODEL
knn_model = KNeighborsClassifier(n_neighbors = 11)
knn_model.fit(train_image_flat,train_label)
knn_acc = knn_model.score(test_image_flat,test_label)
print(f"KNN ACCURACY - {knn_acc}")
knn_pred = knn_model.predict(test_image_flat)

# CNN MODEL
cnn_model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
])
cnn_model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])
cnn_model.fit(train_image,train_label,epochs = 10)
cnn_loss,cnn_acc = cnn_model.evaluate(test_image,test_label)
print(f"CNN ACCURACY - {cnn_acc}")
cnn_pred = cnn_model.predict(test_image)

for i in range(5) :
    find = random.randint(0,len(test_image) - 1)
    plt.imshow(test_image[find],cmap = plt.cm.binary)
    plt.title(f"ACTUAL - {names[test_label[find]]}\nKNN - {names[knn_pred[find]]} | CNN - {names[np.argmax(cnn_pred[find])]}")
    plt.show()