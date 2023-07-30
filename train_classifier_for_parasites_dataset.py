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

import tensorflow as tf
import numpy as np
import os
from src.decision_boundary_mapper.utils.dataReader import import_folder_dataset

CLASSIFIER_NAME = 'classifier'
FOLDER = './data/parasites_focus_plane_divided/ovos/resized_64'
DATASET_NAME = "_".join(FOLDER.split(os.sep)[-2:])
SAVE_FOLDER = os.path.join("tmp", DATASET_NAME, CLASSIFIER_NAME)


if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

(X_train, Y_train), (X_test, Y_test) = import_folder_dataset(FOLDER)

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

num_classes = len(np.unique(Y_train))
print('Number of classes: ', num_classes)

print("\n\nTrain set statistics")
for i in range(num_classes):
    num_classes_train = np.count_nonzero(Y_train == i)
    print(f'Number of {i} class in train set: ', num_classes_train)


print("\n\nTest set statistics")
for i in range(num_classes):
    num_classes_test = np.count_nonzero(Y_test == i)
    print(f'Number of {i} class in test set: ', num_classes_test)

classifier = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
        ], name=CLASSIFIER_NAME)

classifier.compile(optimizer=tf.keras.optimizers.Adam(), 
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

classifier.fit(X_train, Y_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_split=0.2)

classifier.evaluate(X_test, Y_test)

classifier.save(SAVE_FOLDER, save_format="tf")
