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
import pandas as pd
from PIL import Image

from .. import Logger

ALLOWED_DATA_FORMATS = [".csv", ".txt", ".npy"]


def import_mnist_dataset() -> tuple:
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


def import_fashion_mnist_dataset() -> tuple:
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


def import_cifar10_dataset() -> tuple:
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

def import_folder_dataset(dir:str, train_test_split = 0.8) -> tuple:
    console = Logger(name="Dataset importer")
    
    file_list = [os.path.join(dir, file) for file in os.listdir(dir)]
    Y = np.array([int(fname.split("_")[0]) - 1 for fname in os.listdir(dir)]) 
    X = np.array([np.array(Image.open(fname)) for fname in file_list])
    console.log(f"Dataset imported from {dir}")
    console.log(f"Dataset shape: X {X.shape}, Y {Y.shape}")
    
    size = len(Y)
    #np.random.seed(42)
    #indices = np.random.permutation(X.shape[0])
    #X, Y = X[indices], Y[indices]
    (train_X, test_X) = X[:int(train_test_split*size)], X[int(train_test_split*size):]
    (train_Y, test_Y) = Y[:int(train_test_split*size)], Y[int(train_test_split*size):]
    
    console.log(f"Train set: {train_X.shape}")
    console.log(f"Test set: {test_X.shape}")
    return (train_X, train_Y), (test_X, test_Y)

    
def import_dataset(file_path:str,
                   labels_index: int | None = 0,
                   limit: int | None = None,
                   shape: tuple | None = None,
                   ) -> tuple:       
    """Imports a dataset from a file"""
    if not os.path.exists(file_path):
        print("File not found")
        return None, None
    
    if not os.path.splitext(file_path)[-1] in ALLOWED_DATA_FORMATS:
        print("File format not supported, please use one of the following: ", ALLOWED_DATA_FORMATS)
        return None, None
     
    data = None   
    if os.path.splitext(file_path)[-1] == ".csv" or os.path.splitext(file_path)[-1] == ".txt":
        data = import_csv_dataset(file_path, limit=limit)

    if os.path.splitext(file_path)[-1] == ".npy":
        data = import_npy_dataset(file_path, limit=limit)
    
    if data is None:
        return None, None
    
    if labels_index is None:
        return data, None
 
    
    Y = data[:, labels_index]
    X = np.delete(data, labels_index, axis=1)
      
    if shape is not None:
        X = X.reshape(X.shape[0], *shape)
    return X, Y
  
def import_csv_dataset(file_path: str,
                       limit: int | None = None) -> np.ndarray:
    """Imports a dataset from a csv file

    Args:
        file_path (str): The file path
        limit (_type_, optional): The limit of data points to be loaded. Defaults to None.
    Returns:
        X (np.ndarray): The data points and labels
    """
    data_frame = pd.read_csv(file_path, nrows=limit, skiprows = 1, header=None)
    return np.array(data_frame.values)
   

def import_npy_dataset(file_path: str,
                        limit: int | None = None) -> np.ndarray:
    """Imports a dataset from a npy file"""
    with open(file_path, 'rb') as f:
        data = np.load(f)
    X = data[:limit]
    return X
