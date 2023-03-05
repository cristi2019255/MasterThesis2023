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

from src.decision_boundary_mapper import DBM
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.decision_boundary_mapper.utils.dataReader import import_mnist_dataset



def import_data():
    # import the dataset
    (X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    
    # if needed perform some preprocessing
    SAMPLES_LIMIT = 5000
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255    
    X_train, Y_train = X_train[:int(0.7*SAMPLES_LIMIT)], Y_train[:int(0.7*SAMPLES_LIMIT)]
    X_test, Y_test = X_test[:int(0.3*SAMPLES_LIMIT)], Y_test[:int(0.3*SAMPLES_LIMIT)]
    return X_train, X_test, Y_train, Y_test

def import_classifier():
    # upload a classifier
    # !!! the classifier must be a tf.keras.models.Model !!!
    # !!! change the next line with the path to your classifier !!!
    classifier_path = os.path.join("tmp", "MNIST", "classifier")
    classifier = tf.keras.models.load_model(classifier_path)
    return classifier

def import_2d_data():
    # upload the 2D projection of the data
    with open(os.path.join("tmp", "MNIST", "DBM", "t-SNE", "train_2d.npy"), "rb") as f:
        X2d_train = np.load(f)
    with open(os.path.join("tmp", "MNIST", "DBM", "t-SNE", "test_2d.npy"), "rb") as f:
        X2d_test = np.load(f)
    return X2d_train, X2d_test


def test():
    X_train, X_test, Y_train, Y_test = import_data()
    classifier = import_classifier()
    X2d_train, X2d_test = import_2d_data()
    dbm = DBM(classifier)
    resolution = 288
    
    """
    dbm_info = dbm.generate_boundary_map(X_train, Y_train, 
                                        X_test, Y_test,
                                        X2d_train,
                                        X2d_test ,
                                        resolution=resolution,
                                        use_fast_decoding=False,
                                        load_folder=os.path.join("tmp", "MNIST", "DBM"),
                                        projection='t-SNE')
    """
    dbm_info = dbm.generate_boundary_map(X_train, Y_train, 
                                        X_test, Y_test,
                                        X2d_train,
                                        X2d_test ,
                                        resolution=resolution,
                                        use_fast_decoding=True,
                                        load_folder=os.path.join("tmp", "MNIST", "DBM"),
                                        projection='t-SNE')
    
    img, img_confidence, _,_,_,_ = dbm_info
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    img_confidence = img_confidence / np.max(img_confidence)
    ax.imshow(img, alpha=img_confidence)
    plt.show()
    
    
test()