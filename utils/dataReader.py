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
from Logger.Logger import Logger
from keras.datasets import mnist
import os

def import_mnist_dataset():
    console = Logger(name="MNIST dataset importer")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    console.log("MNIST dataset imported")
    console.log(f"Train set: {train_X.shape}")
    console.log(f"Test set: {test_X.shape}")
    return (train_X, train_y), (test_X, test_y)

def import_csv_dataset(file_path, labels_index = 0, headers = False, separator=",", limit = None, shape = (28, 28)):        
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