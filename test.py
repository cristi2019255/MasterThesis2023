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
                    if abs(img1[i,j] - img2[i,j]) > 0.03:
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
    resolution = 600
    
    dbm.generate_boundary_map(X_train, Y_train, 
                                        X_test, Y_test,
                                        X2d_train,
                                        X2d_test ,
                                        resolution=10,
                                        use_fast_decoding=False,
                                        load_folder=os.path.join("tmp", "MNIST", "DBM"),
                                        projection='t-SNE')
    
    start = time.time()
    img1, img_confidence1 = dbm._get_img_dbm_fast_(resolution)
    end = time.time()
    print("Fast decoding time: ", end - start)
    
    with open("img1.npy", "wb") as f:
        np.save(f, img1)
    with open("img_confidence1.npy", "wb") as f:
        np.save(f, img_confidence1)
    
    
    start = time.time()      
    img2, img_confidence2 = dbm._get_img_dbm_(resolution)
    end = time.time()  
    print("Slow decoding time: ", end - start)
    
    with open("img2.npy", "wb") as f:
        np.save(f, img2)
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
        
    print("Comparing dbm images...")
    compare_images(img1, img2)
    print("Comparing confidence images...")
    compare_images(img_confidence1, img_confidence2, comparing_confidence=True)


def test3():
    X_train, X_test, Y_train, Y_test = import_data()
    classifier = import_classifier()
    X2d_train, X2d_test = import_2d_data()
    dbm = DBM(classifier)
    resolution = 256
    
    dbm.generate_boundary_map(X_train, Y_train, 
                                        X_test, Y_test,
                                        X2d_train,
                                        X2d_test ,
                                        resolution=10,
                                        use_fast_decoding=False,
                                        load_folder=os.path.join("tmp", "MNIST", "DBM"),
                                        projection='t-SNE')
    
    start = time.time()
    errors = dbm.generate_inverse_projection_errors(resolution=500)
    with open("inverse_projection_errors_big.npy", "wb") as f:
        np.save(f, errors)
    end = time.time()
    print("Time: ", end - start)
    
def show_errors():
    with open("img_confidence2.npy", "rb") as f:
        errors = np.load(f)
    
    with open("img_confidence1.npy", "rb") as f:
        errors2 = np.load(f)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(errors)
    ax2.imshow(errors2)
    
    errors_count = 0
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            if abs(errors[i,j] - errors2[i,j]) > 0.03:
                ax1.plot(j, i, 'ro')
                errors_count += 1
            if errors[i,j] == errors2[i,j]:
                ax2.plot(j, i, 'go')
    
    print(errors_count)
    
    plt.show()

from scipy import interpolate
def test_interpolation():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    resolution = 10
    sparse_map = [(0, 0, 0), (2, 1, 1), (0, 3, 2), (4, 1, 3), (4, 4, 4), (0, 4, 5), (4, 0, 6), (2, 2, 7), (2, 3, 8), (3, 2, 9), (3, 3, 10), (8,8,10)]
    X, Y, Z = [], [], []
    for (x, y, z) in sparse_map:
        X.append(x)
        Y.append(y)
        Z.append(z)
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    xi = np.linspace(0, resolution-1, resolution)
    yi = np.linspace(0, resolution-1, resolution)
    grid = interpolate.griddata((X, Y), Z, (xi[None,:], yi[:,None]), method="cubic")

    print(grid)

    iterpolator = interpolate.SmoothBivariateSpline(X, Y, Z, kx=2, ky=2)
    z = iterpolator(xi[None,:], yi[:,None])

    print(z)    
    
    grid = np.array(grid)
    z = np.array(z)
    
    grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
    z = (z - np.min(z)) / (np.max(z) - np.min(z))
    
    
    ax1.imshow(grid)
    ax2.imshow(z)
    plt.show()

#test()
#test2()
#test3()
test_interpolation()