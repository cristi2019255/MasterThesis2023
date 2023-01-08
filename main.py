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
from utils.DBM import DBM

from utils.opf import opf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # uncomment to disable GPU, run on CPU only 
import tensorflow as tf 

import numpy as np
from utils.transformer import transform_mnist_to_tf_dataset
from utils.reader import import_mnist_dataset, load_mnist_preprocessed
from models.VGG16 import load_convolution_corpus, train
from models.tools import load_model, predict
from utils.tsneTools import load_tsne_embedding, plot_tsne_embedding
from tensorflow.keras.utils import to_categorical


SAMPLES_LIMIT = 5000

def d(X_train, Y_train, X_test, Y_test):
    # DBM - decision boundary mapping
    # Input: X_train, Y_train, X_test, Y_test - MNIST dataset
    # Output: A 2D image (resolution x resolution) where each pixel represents a class 
    
    # Steps: 
    # 1. Train an autoencoder on the training data (this will be used to reduce the dimensionality of the data) nD -> 2D
    # 2. For each pixel in the 2D image, predict the class of the pixel using the trained autoencoder, the decoder part
    # 3. Generate the 2D image
    # 4. Plot the 2D image
    # 5. Enjoy!
    pass
    
def main():
    # import MNIST dataset
    #(X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    
    # load preprocessed dataset
    (X_train, Y_train), (X_test, Y_test) = load_mnist_preprocessed()
    
    # limiting to first 1000 samples for testing
    X_train = X_train[:int(0.7 * SAMPLES_LIMIT)]
    Y_train = Y_train[:int(0.7 * SAMPLES_LIMIT)]
    X_test = X_test[:int(0.3 * SAMPLES_LIMIT)]
    Y_test = Y_test[:int(0.3 * SAMPLES_LIMIT)]
    
    #(X_train, Y_train), (X_test, Y_test) = transform_mnist_to_tf_dataset(X_train, Y_train, X_test, Y_test)
    
    
    model = load_model("MNIST")
    # model, history = train(X_train, Y_train, X_test, Y_test)
    
    #predict(X_test, Y_test, model)
    
    # corpus = get_convolution_corpus(X_train, model)
    # corpus = load_convolution_corpus(model.name)
    labels = np.array([np.argmax(y) for y in Y_train])
    #plot_tsne_embedding(labels, model.name)
    X = load_tsne_embedding(model.name)
    Y = labels
    Y = opf(X, Y)
    print(Y.shape)
    Y_train = to_categorical(Y)
    model, history = train(X_train, Y_train, X_test, Y_test)
    

def main_dbm():
    # import MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    
    # load preprocessed dataset
    #(X_train, Y_train), (X_test, Y_test) = load_mnist_preprocessed()
    
    # limiting to first 1000 samples for testing
    X_train = X_train[:int(0.7 * SAMPLES_LIMIT)]
    Y_train = Y_train[:int(0.7 * SAMPLES_LIMIT)]
    X_test = X_test[:int(0.3 * SAMPLES_LIMIT)]
    Y_test = Y_test[:int(0.3 * SAMPLES_LIMIT)]
    
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    
    num_classes = np.unique(Y_train).shape[0]
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])
    
    dbm = DBM(classifier)
    dbm.generate_boundary_map(X_train, Y_train, X_test, Y_test, 
                              train_epochs=10, 
                              train_batch_size=128,
                              resolution=256)

if __name__ == '__main__':
    main_dbm()
        