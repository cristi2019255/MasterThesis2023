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
from keras.datasets import fashion_mnist, mnist, cifar10
import os

from .. import Logger

def import_mnist_dataset():
    """Imports the MNIST dataset from keras.datasets.mnist

    Returns:
        (X_train, Y_train), (X_test, Y_test): The train and test sets
    """
    console = Logger(name="MNIST dataset importer")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    console.log("MNIST dataset imported")
    console.log(f"Train set: {train_X.shape}")
    console.log(f"Test set: {test_X.shape}")
    return (train_X, train_y), (test_X, test_y)


def import_fashion_mnist_dataset():
    """Imports the FASHION MNIST dataset from keras.datasets.fashion_mnist

    Returns:
        (X_train, Y_train), (X_test, Y_test): The train and test sets
    """
    console = Logger(name="FASHION MNIST dataset importer")
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    console.log("FASHION MNIST dataset imported")
    console.log(f"Train set: {train_X.shape}")
    console.log(f"Test set: {test_X.shape}")
    return (train_X, train_y), (test_X, test_y)

def import_cifar10_dataset():
    """Imports the CIFAR10 dataset from keras.datasets.cifar10
    
    Returns:
        (X_train, Y_train), (X_test, Y_test): The train and test sets
    """
    console = Logger(name="CIFAR10 dataset importer")    
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    console.log("CIFAR10 dataset imported")
    console.log(f"Train set: {train_X.shape}")
    console.log(f"Test set: {test_X.shape}")
    return (train_X, train_y), (test_X, test_y)


def import_csv_dataset(file_path:str, labels_index:int = 0, headers:bool = False, 
                       separator:str=",", limit:int = None, shape:tuple = (28, 28)):        
    """Imports a dataset from a csv file

    Args:
        file_path (str): The file path
        labels_index (int, optional): The index of the column with the data labels. Defaults to 0.
        headers (bool, optional): If headers are present in the file set to True. Defaults to False.
        separator (str, optional): The file data separator. Defaults to ",".
        limit (_type_, optional): The limit of data points to be loaded. Defaults to None.
        shape (tuple, optional): The data points shape. Defaults to (28, 28).

    Returns:
        X, Y (np.ndarray, np.ndarray): The data points and labels
    """
    
    if not os.path.exists(file_path):
        print("File not found")
        return None, None
    
    if not (file_path.endswith(".csv") or file_path.endswith(".txt")):
        print("File format not supported")
        return None, None
    
    try:
        with open(file_path, "r") as f:
            if headers:
                f.readline()
            lines = f.readlines()
            
            if limit is not None and limit < len(lines):
                lines = lines[:limit]
            
            lines = [line.strip().split(separator) for line in lines]
            X = [line[:labels_index] + line[labels_index+1:] for line in lines]
            Y = [line[labels_index] for line in lines]
        
        X, Y = np.array(X).astype("float32"), np.array(Y).astype("int")    
        X = X.reshape(X.shape[0], *shape)
        X /= 255
        return X, Y
    except Exception as e:
        print(e)
        return None, None    