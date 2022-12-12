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

from models.tools import print_model_summary
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from utils.Logger import Logger

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