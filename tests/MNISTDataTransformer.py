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

import os
import numpy as np
from keras.utils.image_utils import img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical

from src.Logger import Logger

MNIST_DATA_FOLDER = os.path.join(os.getcwd(), "data", "MNIST")

def transform_mnist_to_tf_dataset(X_train, Y_train, X_test, Y_test):
    console = Logger(name="MNIST dataset transformer")
    
    # reshape data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    console.log(f"Reshaped to 28x28 representation {X_train.shape}")

    # Convert the images into 3 channels as MNIST images are Black and White so have 1 channel
    X_train = np.array([np.dstack((img, img, img)) for img in X_train])
    X_test = np.array([np.dstack((img, img, img)) for img in X_test])
    console.log(f"Transformed image data to 3 channels: {X_train.shape}")
    
    
    # Resize the images 48*48 as required by VGG16
    X_train = np.asarray([img_to_array(array_to_img(img, scale=False).resize((48,48))) for img in X_train])
    X_test = np.asarray([img_to_array(array_to_img(img, scale=False).resize((48,48))) for img in X_test])
    console.log(f"Resized images to 48x48 representation {X_train.shape}")

    # change to float datatype
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalize to range 0-1
    X_train /= 255
    X_test /= 255
    console.log("Normalized images to 0-1 representation")
    
    # one-hot encode target column
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    console.log("One-hot encoded target column")
   
    console.log("MNIST dataset transformed")
    
    # Save the transformed dataset
    with open(os.path.join(MNIST_DATA_FOLDER, "mnist_X_train.npy"), "wb") as f:
        np.save(f, X_train)
    with open(os.path.join(MNIST_DATA_FOLDER,"mnist_X_test.npy"), "wb") as f:
        np.save(f, X_test)
    with open(os.path.join(MNIST_DATA_FOLDER, "mnist_Y_train.npy"), "wb") as f:
        np.save(f, Y_train)
    with open(os.path.join(MNIST_DATA_FOLDER, "mnist_Y_test.npy"), "wb") as f:
        np.save(f, Y_test)
    
    return (X_train, Y_train), (X_test, Y_test)