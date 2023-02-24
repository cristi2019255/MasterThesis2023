# Copyright 2023 Cristian Grosu
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

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from math import sqrt
from tensorflow.keras.applications import VGG16

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

tf.random.set_seed(42)
tf.keras.backend.clear_session()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train[:3000]
y_train = y_train[:3000]
X_test = X_test[:1000]
y_test = y_test[:1000]


X_train = rgb2gray(X_train)
X_test = rgb2gray(X_test)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#print(X_train.shape)
#plt.imshow(X_train[0], cmap='gray')
#plt.show()

output_shape = (32, 32)
output_size = output_shape[0] * output_shape[1]


encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=output_shape),    
    tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01), kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
    tf.keras.layers.Dense(32, activation='sigmoid', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)), 
    #tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)), 
    #tf.keras.layers.Dense(16, activation='sigmoid', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)), 
    #tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),  
    #tf.keras.layers.Dense(2, activation='linear', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),    
], name="encoder")

decoder = tf.keras.Sequential([
    #tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)), 
    #tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)), 
    #tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)), 
    #tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),      
    tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
    tf.keras.layers.Dense(output_size, activation='sigmoid', kernel_initializer='he_uniform'),
    tf.keras.layers.Reshape(output_shape)
], name="decoder")

autoencoder = tf.keras.Sequential([encoder, decoder], name="autoencoder")
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

autoencoder.fit(X_train, X_train, 
                epochs=100, 
                batch_size=128, 
                callbacks=[early_stopping],
                validation_data=(X_test, X_test))

autoencoder.save("autoencoder.h5")


autoencoder = tf.keras.models.load_model("autoencoder.h5")

predictions = autoencoder.predict(X_train)
print(predictions.shape)


def show_predictions(dataNd: np.ndarray, predictions: np.ndarray, n_samples:int = 32):
        
        plt.figure(figsize=(20, 10))
        m = int(sqrt(n_samples))
        n = m * m
        
        for i in range(1,2*n,2):
            plt.subplot(m, 2*m, i)
            plt.imshow(dataNd[i], cmap='gray')
            plt.axis('off')
            plt.subplot(m, 2*m, i+1)
            plt.imshow(predictions[i], cmap='gray')
            plt.axis('off')
        
        plt.show()

show_predictions(X_train, predictions)

"""
classifier = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32)),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01))
]);
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier.fit(predictions, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test))
classifier.save("classifier.h5")
classifier.evaluate(X_test, y_test)
"""