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

import tensorflow as tf
from utils.Logger import Logger
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

MODELS_FOLDER = os.path.join("models", "model")


def freeze_layers(model, layers_to_freeze=1):
    
    assert layers_to_freeze < len(model.layers), "Number of layers to freeze is smaller than the number of layers in the model."
    
    console = Logger(name="Model layers freezer")   
    console.log("Freezing layers of the model.\n")
    
    for layer in model.layers[:-layers_to_freeze]:
        layer.trainable=False
    console.success("Layers of the model frozen.\n")
    
    return model

def print_model_summary(model, console = None):
    if console is None:
        console = Logger(name="VGG-16 model summary printer")
    
    console.log("Summary of the model.\n")
    model.summary() 

def plot(model):
    dir = os.path.join(MODELS_FOLDER, model.name)
    if not os.path.exists(os.path.dirname(dir)):
        os.makedirs(os.path.dirname(dir))
    file = os.path.join(dir, f"{model.name}.png")
    
    plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)    

def load_model(model_name):
    file = os.path.join(MODELS_FOLDER, model_name, f"{model_name}.h5")
    return tf.keras.models.load_model(file)

def save_model(model):
    dir = os.path.join(MODELS_FOLDER, model.name)
    if not os.path.exists(os.path.dirname(dir)):
        os.makedirs(os.path.dirname(dir))
    file = os.path.join(dir, f"{model.name}.h5")
    model.save(file)
    
def save_history(history, model_name):
    dir = os.path.join(MODELS_FOLDER, model_name)
    if not os.path.exists(os.path.dirname(dir)):
        os.makedirs(os.path.dirname(dir))
    file = os.path.join(dir, f"{model_name}_history.npy")
    with open(file, "wb") as f:
        np.save(f, history.history)
    
def load_history(model_name):
    dir = os.path.join(MODELS_FOLDER, model_name)
    file = os.path.join(dir, f"{model_name}_history.npy")
    with open(file, "rb") as f:
        history = np.load(f, allow_pickle=True).item()
    return history

def plot_history(model_name, show=True):
    dir = os.path.join(MODELS_FOLDER, model_name)
    
    history = load_history(model_name)
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()
    plt.savefig(os.path.join(dir, f"{model_name}_history_ACC.png"))
    

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')

    plt.legend()
    plt.savefig(os.path.join(dir, f"{model_name}_history_LOSS.png"))
    
    if show:
        plt.show()
        
    

def predict(X_test, Y_test, model):
    # TODO: update this function to work better
    console = Logger(name="Model predictor")
    plt.figure(figsize=(20,10))
    
    x = X_test[:15]
    y = Y_test[:15]
    predictions=model.predict(x)
    for i in range(15):
        plt.subplot(3,5,i+1)
        plt.axis('off')     
        plt.imshow(x[i])
        pred = predictions[i]
        output=np.argmax(pred)
        console.success(f"Predicted probabilities are: {pred}")
        console.success(f"Predicted digit is: {output}")
        plt.title(f"Predicted: {output} ({pred[output]*100:.2f}%) \n Actual: {np.argmax(y[i])}")    
        
    plt.show()
    