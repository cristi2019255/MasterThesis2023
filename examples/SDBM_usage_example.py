# Copyright 2023 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils import import_mnist_dataset
import tensorflow as tf
import numpy as np
from DBM import SDBM
import matplotlib.pyplot as plt

def SDBM_usage_example():
        
    # import the dataset
    (X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    
    # if needed perform some preprocessing
    SAMPLES_LIMIT = 5000
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train, Y_train = X_train[:int(0.7*SAMPLES_LIMIT)], Y_train[:int(0.7*SAMPLES_LIMIT)]
    X_test, Y_test = X_test[:int(0.3*SAMPLES_LIMIT)], Y_test[:int(0.3*SAMPLES_LIMIT)]
    
    # get the number of classes
    num_classes = np.unique(Y_train).shape[0]
    
    # create a classifier
    classifier = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])
    
    # create the DBM
    sdbm = SDBM(classifier=classifier)
    
    # use the SDBM to get the decision boundary map
    img, img_confidence, _, _ = sdbm.generate_boundary_map(X_train, Y_train, 
                                                                X_test, Y_test, 
                                                                resolution=256)
    
    
    # make the decision boundary map pretty, by adding the colors and the confidence
    COLORS_MAPPER = {
        -2: [0,0,0], # setting original test data to black
        -1: [1,1,1], # setting original train data to white
        0: [1,0,0], 
        1: [0,1,0], 
        2: [0,0,1], 
        3: [1,1,0], 
        4: [0,1,1], 
        5: [1,0,1], 
        6: [0.5,0.5,0.5], 
        7: [0.5,0,0], 
        8: [0,0.5,0], 
        9: [0,0,0.5]
    }

    color_img = np.zeros((img.shape[0], img.shape[1], 4))
    for i, j in np.ndindex(img.shape):       
        color_img[i,j] = COLORS_MAPPER[img[i,j]] + [img_confidence[i,j]]
    
    # plot the decision boundary map
    plt.title("Decision boundary map")
    plt.axis("off")
    plt.imshow(color_img)
    plt.show()
    
    # use the SDBM to get the inverse projection errors
    img_inverse_projection_errors = sdbm.generate_inverse_projection_errors()
    # plot the inverse projection errors
    fig, ax = plt.subplots()
    ax.set_title("Inverse projection errors")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_ax = ax.imshow(img_inverse_projection_errors, cmap="Reds")
    fig.colorbar(img_ax, ax=ax)
    plt.show()
    
    # use the SDBM to get the projection errors
    img_projection_errors = sdbm.generate_projection_errors()
    # plot the projection errors
    fig, ax = plt.subplots()
    ax.set_title("Projection errors")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_ax = ax.imshow(img_projection_errors, cmap="Reds")
    fig.colorbar(img_ax, ax=ax)
    plt.show()
    