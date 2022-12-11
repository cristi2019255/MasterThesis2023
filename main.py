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
import matplotlib.pyplot as plt

import os
import random
from tqdm import tqdm # for progress bar

# Libraries for TensorFlow
from tensorflow.keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from tensorflow import keras

# Library for Transfer Learning
from keras.applications.vgg16 import preprocess_input

from utils.transformer import transform_mnist_to_tf_dataset
from utils.reader import import_mnist_dataset, load_mnist_preprocessed
from models.VGG16 import build_model, build_custom_model, compile_model
from models.tools import freeze_layers, load_model, plot, plot_history, save_history, save_model

EPOCHS = 16
BATCH_SIZE = 128
MODELS_FOLDER = os.path.join("models", "model")

def predict(X_test, Y_test, model):
    # TODO: update this function to work better
    x = X_test[:1]
    plt.imshow(x[0])
    plt.show()
    prediction=model.predict(x)
    output=np.argmax(prediction)
    print("Predicted probabilities are: ", prediction)
    print("Predicted digit is: ", output)

def train(X_train, Y_train, X_test, Y_test):
    model = build_custom_model()
    model = freeze_layers(model, layers_to_freeze=2)
    model = compile_model(model)
    plot(model)
    
    filepath = os.path.join(MODELS_FOLDER, model.name, f"{model.name}.h5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    
    # Fit the Model
    history = model.fit(X_train,
                        Y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=True,
                        validation_data=(X_test,Y_test)
                        )
    save_model(model)
    save_history(history, model.name)
    plot_history(model.name)
    
    return model, history

def main():
    # import MNIST dataset
    #(X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    #(X_train, Y_train), (X_test, Y_test) = transform_mnist_to_tf_dataset(X_train, Y_train, X_test, Y_test)
    
    # load preprocessed dataset
    (X_train, Y_train), (X_test, Y_test) = load_mnist_preprocessed()
    
    # limiting to first 1000 samples for testing
    X_train = X_train[:700]
    Y_train = Y_train[:700]
    X_test = X_test[:300]
    Y_test = Y_test[:300]
    
    model = load_model("MNIST")
    #model, history = train(X_train, Y_train, X_test, Y_test)
    
    predict(X_test, Y_test, model)
    

if __name__ == '__main__':
    main()
    