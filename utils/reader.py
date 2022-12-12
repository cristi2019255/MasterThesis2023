# Copyright 2022 Cristian Grosu
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

import numpy as np
from utils.Logger import Logger
from keras.datasets import mnist
import os

MNIST_DATA_FOLDER = os.path.join(os.getcwd(), "data", "MNIST")

def import_mnist_dataset():
    console = Logger(name="MNIST dataset importer")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    console.log("MNIST dataset imported")
    console.log(f"Train set: {train_X.shape}")
    console.log(f"Test set: {test_X.shape}")
    return (train_X, train_y), (test_X, test_y)

def load_mnist_preprocessed():
    # Load the preprocessed dataset
    with open(os.path.join(MNIST_DATA_FOLDER, "mnist_X_train.npy"), "rb") as f:
        X_train = np.load(f)
    with open(os.path.join(MNIST_DATA_FOLDER, "mnist_X_test.npy"), "rb") as f:
        X_test = np.load(f)
    with open(os.path.join(MNIST_DATA_FOLDER, "mnist_Y_train.npy"), "rb") as f:
        Y_train = np.load(f)
    with open(os.path.join(MNIST_DATA_FOLDER, "mnist_Y_test.npy"), "rb") as f:
        Y_test = np.load(f)
    return (X_train, Y_train), (X_test, Y_test)
