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
from models.tools import freeze_layers, print_model_summary
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from utils.Logger import Logger
from tensorflow.keras.models import Model
from models.tools import freeze_layers, load_model, plot, plot_history, predict, save_history, save_model
from tensorflow.keras.callbacks import ModelCheckpoint


EPOCHS = 16
BATCH_SIZE = 100
MODELS_FOLDER = os.path.join("models", "model")

def build_model(model_name = "MNIST"):
    console = Logger(name="VGG-16 model builder")   
     
    # initializing model with weights='imagenet'i.e. we are carring its original weights
    console.log("Initializing VGG16 model with weights='imagenet'.\n")
    model_vgg16=VGG16(name = model_name, weights='imagenet')
    
    return model_vgg16

def build_custom_model(input_shape=(48,48,3), include_top=False, output_shape=10, model_name = "MNIST"):
    console = Logger(name="VGG-16 model builder")   

    input_layer=layers.Input(shape=input_shape)
     
    
    # initializing model with weights='imagenet'i.e. we are carring its original weights
    console.log(f"Initializing VGG16 model with weights='imagenet', input_tensor = {input_shape}, include_top = {include_top}.\n")
    model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=include_top)

    last_layer=model_vgg16.output # we are taking last layer of the model

    # Add flatten layer: we are extending Neural Network by adding flattn layer
    flatten=layers.Flatten()(last_layer) 

    # Add dense layer to the final output layer
    output_layer=layers.Dense(output_shape,activation='softmax')(flatten)

    # Creating a model with input and output layer
    model = models.Model(name=model_name,inputs=input_layer,outputs=output_layer)
    
    return model

def compile_model(model, loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'], summary=True):
    console = Logger(name="VGG-16 model compiler")
    # Compiling Model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    console.success("Model compilation completed.")
    
    if summary:
        print_model_summary(model, console)
    
    return model

def train(X_train, Y_train, X_test, Y_test, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model = build_custom_model()
    model = freeze_layers(model, layers_to_freeze=2)
    model = compile_model(model)
    plot(model)
    
    filepath = os.path.join(MODELS_FOLDER, model.name, f"{model.name}.h5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=1)
    
    # Fit the Model
    history = model.fit(X_train,
                        Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=True,
                        validation_data=(X_test,Y_test)
                        )
    save_model(model)
    save_history(history, model.name)
    plot_history(model.name)
    
    return model, history


def get_convolution_corpus(X_train, model):
    console = Logger("Get convolution corpus")
    # remove last two layers
    model1 = Model(inputs=model.input, outputs=model.layers[-3].output)
    model1 = compile_model(model1, summary=False)
    # get convolution corpus
    console.log("Getting convolution corpus...")
    convolution_corpus = model1.predict(X_train)
    console.log(f"Convolution corpus shape: {convolution_corpus.shape}")    
    # returns a tensor of shape (batch_size, 1, 1, 512)
    # reshape to (batch_size, 512)
    convolution_corpus = convolution_corpus.reshape(convolution_corpus.shape[0], convolution_corpus.shape[3])
    console.log(f"Reshaping convolution corpus ... The new shape is: {convolution_corpus.shape}")
    
    # save convolution corpus
    if not os.path.exists(os.path.join("data", model.name)):
        os.makedirs(os.path.join("data", model.name))
    with open(os.path.join("data", model.name, "convolution_corpus.npy"), "wb") as f:
        np.save(f, convolution_corpus)
    
    return convolution_corpus


def load_convolution_corpus(model_name):
    with open(os.path.join("data", model_name, "convolution_corpus.npy"), "rb") as f:
        convolution_corpus = np.load(f)
    return convolution_corpus