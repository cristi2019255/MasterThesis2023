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

from keras.datasets import fashion_mnist, mnist, cifar10
import tensorflow as tf
import os

SAMPLES_LIMIT = 5000
CLASSIFIER_NAME = "classifier"

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
        
X_train, Y_train = X_train[:int(0.7*SAMPLES_LIMIT)], Y_train[:int(0.7*SAMPLES_LIMIT)]
X_test, Y_test = X_test[:int(0.3*SAMPLES_LIMIT)], Y_test[:int(0.3*SAMPLES_LIMIT)]

num_classes = 10

classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
], name=CLASSIFIER_NAME) 
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, Y_train, epochs=5, batch_size=32)

save_path = os.path.join("tmp", "CIFAR10", "classifier")
classifier.save(save_path, save_format="tf")
