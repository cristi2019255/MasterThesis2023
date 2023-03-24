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

import tensorflow as tf
import numpy as np
import os
import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SemiSupervisedOPF

from ..utils import import_mnist_dataset

def import_data():
    # import the dataset
    (X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    
    # if needed perform some preprocessing
    SAMPLES_LIMIT = 5000
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255    
    X_train, Y_train = X_train[:int(0.7*SAMPLES_LIMIT)], Y_train[:int(0.7*SAMPLES_LIMIT)]
    X_test, Y_test = X_test[:int(0.3*SAMPLES_LIMIT)], Y_test[:int(0.3*SAMPLES_LIMIT)]
    return X_train, X_test, Y_train, Y_test

def import_classifier():
    # upload a classifier
    # !!! the classifier must be a tf.keras.models.Model !!!
    # !!! change the next line with the path to your classifier !!!
    classifier_path = os.path.join("tmp", "MNIST", "classifier")
    classifier = tf.keras.models.load_model(classifier_path)
    return classifier

def generate_classifier(X_train, Y_train):    
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(Y_train))    
    
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax),
    ])
    
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, Y_train, epochs=20)
    return classifier
    

def opf(X_train, Y_train):
    pr_x = int(0.1*len(X_train))
    pr_y = int(0.1*len(Y_train))
    X_train, X_unlabeled, Y_train, Y_unlabeled = X_train[:pr_x], X_train[pr_x:], Y_train[:pr_y], Y_train[pr_y:]

    # Creates a SemiSupervisedOPF instance
    opf = SemiSupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

    # Fits training data along with unlabeled data into the semi-supervised classifier
    opf.fit(X_train, Y_train, X_unlabeled)
    
    # Predicts new data
    preds = opf.predict(X_unlabeled)
    
    # Calculating accuracy
    acc = g.opf_accuracy(Y_unlabeled, preds)
    print(f"OPF Accuracy: {acc}")

    return np.hstack((Y_train, preds))


def import_2d_data():
    # upload the 2D projection of the data
    with open(os.path.join("tmp", "MNIST", "DBM", "t-SNE", "train_2d.npy"), "rb") as f:
        X2d_train = np.load(f)
    with open(os.path.join("tmp", "MNIST", "DBM", "t-SNE", "test_2d.npy"), "rb") as f:
        X2d_test = np.load(f)
    return X2d_train, X2d_test

