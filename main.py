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

from utils.opf import opf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # uncomment to disable GPU, run on CPU only 
import tensorflow as tf 
from keras import backend as K

import numpy as np
from utils.transformer import transform_mnist_to_tf_dataset
from utils.reader import import_mnist_dataset, load_mnist_preprocessed
from models.VGG16 import load_convolution_corpus, train
from models.tools import load_model, predict
from utils.tsneTools import load_tsne_embedding, plot_tsne_embedding
from tensorflow.keras.utils import to_categorical

SAMPLES_LIMIT = 5000

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
    
    
if __name__ == '__main__':
    # TODO: check GPU errors
    main()
        