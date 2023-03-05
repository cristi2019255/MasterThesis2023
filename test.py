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

import time
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

def compare_images(img1, img2, comparing_confidence=False):
    errors = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j] != img2[i,j]:
                if comparing_confidence:
                    if abs(img1[i,j] - img2[i,j]) > 0.01:
                        errors += 1
                else:
                    errors += 1
    
    print("Errors: ", errors)
    print("Error rate: ", errors / (img1.shape[0] * img1.shape[1]) * 100, "%")
    return errors

def test():
    X_train, X_test, Y_train, Y_test = import_data()
    classifier = import_classifier()
    X2d_train, X2d_test = import_2d_data()
    dbm = DBM(classifier)
    resolution = 288
    
    dbm.generate_boundary_map(X_train, Y_train, 
                                        X_test, Y_test,
                                        X2d_train,
                                        X2d_test ,
                                        resolution=100,
                                        use_fast_decoding=False,
                                        load_folder=os.path.join("tmp", "MNIST", "DBM"),
                                        projection='t-SNE')
    
    start = time.time()
    img1, img_confidence1, _ = dbm._get_img_dbm_fast_(resolution)
    end = time.time()
    print("Fast decoding time: ", end - start)
    
    start = time.time()      
    img2, img_confidence2, _ = dbm._get_img_dbm_(resolution)
    end = time.time()  
    print("Slow decoding time: ", end - start)
    
    with open("img1.npy", "wb") as f:
        np.save(f, img1)
    with open("img2.npy", "wb") as f:
        np.save(f, img2)
    with open("img_confidence1.npy", "wb") as f:
        np.save(f, img_confidence1)
    with open("img_confidence2.npy", "wb") as f:
        np.save(f, img_confidence2)

def test2(): 
    with open("img1.npy", "rb") as f:
        img1 = np.load(f)
    with open("img2.npy", "rb") as f:
        img2 = np.load(f)
    with open("img_confidence1.npy", "rb") as f:
        img_confidence1 = np.load(f)
    with open("img_confidence2.npy", "rb") as f:
        img_confidence2 = np.load(f)
    
    m = np.max(img_confidence1)
    print(m)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    for i in range(img_confidence1.shape[0]):
        for j in range(img_confidence1.shape[1]):
            if img_confidence1[i,j] > 1:                
                ax1.plot(j,i, 'ro')
                img_confidence1[i,j] = 1
                
    ax1.imshow(img_confidence1, cmap='gray')
    ax2.imshow(img1)
    plt.show()    
    
    #img_confidence1 = img_confidence1 / np.max(img_confidence1)
    
    print("Comparing dbm images...")
    compare_images(img1, img2)
    print("Comparing confidence images...")
    compare_images(img_confidence1, img_confidence2, comparing_confidence=True)

test2()