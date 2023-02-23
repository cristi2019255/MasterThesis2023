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

from src.decision_boundary_mapper.DBM.tools import get_inv_proj_error, get_proj_error
from src.decision_boundary_mapper.utils.dataReader import import_mnist_dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.datasets import mnist
import os
from math import sqrt
from sklearn.neighbors import KDTree
from scipy import interpolate
import time

SAMPLES_LIMIT = 5000

def load_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train, Y_train = X_train[:int(0.7*SAMPLES_LIMIT)], Y_train[:int(0.7*SAMPLES_LIMIT)]
    X_test, Y_test = X_test[:int(0.3*SAMPLES_LIMIT)], Y_test[:int(0.3*SAMPLES_LIMIT)]


    with open("tmp/DBM/t-SNE/train_2d.npy", "rb") as f:
        X_train_2d = np.load(f)
    with open("tmp/DBM/t-SNE/test_2d.npy", "rb") as f:
        X_test_2d = np.load(f)

    return X_train, Y_train, X_test, Y_test, X_train_2d, X_test_2d

def load_model(X_train, Y_train, X_test, Y_test, X_train_2d, X_test_2d):
    if os.path.exists("tmp/DBM/t-SNE/invNN"):
        invNN = tf.keras.models.load_model("tmp/DBM/t-SNE/invNN")
    else:
        print("Building model...")
        output_shape = (28, 28)

        stopping_callback = tf.keras.callbacks.EarlyStopping(verbose=1, min_delta=0.00001, mode='min', patience=20, restore_best_weights=True)

        decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
                tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
                tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
                tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
                tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='sigmoid', kernel_initializer='he_uniform'),
                tf.keras.layers.Reshape(output_shape)
            ], name="decoder")

        classifier = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ], name="classifier")

        input_layer = tf.keras.layers.Input(shape=(2, ))
        decoder_output = decoder(input_layer)
        classifier_output = classifier(decoder_output)
        
        invNN = tf.keras.Model(
            inputs = input_layer,
            outputs = [decoder_output, classifier_output],
            name="invNN")
        
        invNN.compile(optimizer='adam',
                    loss={'decoder': 'mean_squered_error', 'classifier': 'sparse_categorical_crossentropy'},
                    loss_weights={'decoder': 1.0, 'classifier': 0.125},
                    metrics={'decoder': 'accuracy', 'classifier': 'accuracy'},
        )

        print("Training model...")
        hist = invNN.fit(X_train_2d, [X_train, Y_train], 
                            epochs=300, 
                            batch_size=32, 
                            validation_data=(X_test_2d, [X_test, Y_test]),
                            callbacks=[stopping_callback]
                            )
        
        invNN.save("tmp/DBM/invNN", save_format="tf")

        plt.plot(hist.history['loss'], label='loss')
        plt.plot(hist.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

        plt.plot(hist.history['decoder_accuracy'], label='accuracy')
        plt.plot(hist.history['val_decoder_accuracy'], label='val_accuracy')
        plt.legend()
        plt.show()


    return invNN

def show_predictions(invNN, dataNd: np.ndarray, data2d:np.ndarray, labels: np.ndarray, n=16):
        decoded, predicted = invNN.predict(data2d)
        predicted_labels = np.argmax(predicted, axis=1)
        
        plt.figure(figsize=(20, 10))
        m = int(sqrt(n))
        
        for i in range(1,2*n,2):
            plt.subplot(m, 2*m, i)
            plt.imshow(dataNd[i], cmap='gray')
            plt.title(f"Actual: {labels[i]}", color='green' if predicted_labels[i] == labels[i] else 'red', fontsize=12)
            plt.axis('off')
            plt.subplot(m, 2*m, i+1)
            plt.imshow(decoded[i], cmap='gray')
            plt.title(f"Predicted: {predicted_labels[i]}", color='green' if predicted_labels[i] == labels[i] else 'red', fontsize=12)
            plt.axis('off')
        
        plt.show()

def evaluate(invNN, X_test, X_test_2d, Y_test):
    show_predictions(invNN, X_test, X_test_2d, Y_test, n=49)

    score = invNN.evaluate(X_test_2d, [X_test, Y_test])
    print(f"Test loss: {score[0]}")
    print(f"Decoder loss: {score[1]}")
    print(f"Classifier loss: {score[2]}")
    print(f"Decoder accuracy: {score[3]}")
    print(f"Classifier accuracy: {score[4]}")

def get_dbm_img(invNN, X_train_2d, X_test_2d, resolution = 256):
    space2d = np.array([(i / resolution, j / resolution) for i in range(resolution) for j in range(resolution)])
    
    img = np.zeros((resolution, resolution))
    img_confidence = np.zeros((resolution, resolution))

    decoded, predictions = invNN.predict(space2d)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_confidence = np.max(predictions, axis=1)
    spaceNd = decoded.reshape((resolution, resolution, 28, 28))
    img = predicted_labels.reshape((resolution, resolution))
    img_confidence = predicted_confidence.reshape((resolution, resolution))
            
    for [x,y] in X_train_2d:
        i, j = int(x * (resolution - 1)) , int(y * (resolution - 1))
        img[i,j] = -1
        img_confidence[i,j] = 1
    for [x,y] in X_test_2d:
        i, j = int(x * (resolution - 1)) , int(y * (resolution - 1))
        img[i,j] = -2
        img_confidence[i,j] = 1

    return img, img_confidence, spaceNd

def get_inverse_projections_errors(spaceNd, resolution = 256):

    errors = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            errors[i,j] = get_inv_proj_error(i,j, spaceNd)
    
    return errors            
                
def compute_trustworthiness(indices_source, indices_embedded): 

    n = len(indices_source)
    trustworthiness = 0.0
    k = indices_embedded.shape[0]
    
    for j in range(k):

        rank = 0
        while indices_source[rank] != indices_embedded[j]:
            rank += 1

        if rank > k:
            trustworthiness += rank - k

    
    return 2 * trustworthiness / (k * (2*n - 3*k - 1))

def compute_continuity(indices_source, indices_embedded): 

    n = len(indices_embedded)
    continuity = 0.0
    k = indices_source.shape[0]
    
    for j in range(k):

        rank = 0
        while indices_source[j] != indices_embedded[rank]:
            rank += 1

        if rank > k:
            continuity += rank - k

    
    return 2 * continuity / (k * (2*n - 3*k - 1))

def get_continuity_errors(X_train_2d, X_train, resolution = 256):
    proj_errors = np.zeros((resolution, resolution))

    K = 10
    metric = "euclidean"

    X2d = X_train_2d.reshape((X_train_2d.shape[0], -1))
    Xnd = X_train.reshape((X_train.shape[0], -1))

    tree = KDTree(X2d, metric=metric)
    print("Created KDTree for X2d")
    indices_embedded = tree.query(X2d, k=X2d.shape[0], return_distance=False)
    # Drop the actual point itself
    indices_embedded = indices_embedded[:, 1:]
    print("Indices embedded computed")
    tree = KDTree(Xnd, metric=metric)
    print("Created KDTree for Xnd")
    indices_source = tree.query(Xnd, k=K, return_distance=False)
    # Drop the actual point itself
    indices_source = indices_source[:, 1:]
    print("Indices source computed")   
    print(indices_source.shape)

    sparse_map = []

    for k in range(X2d.shape[0]):
        i, j = X2d[k]
        i, j = int(i * (resolution - 1)) , int(j * (resolution - 1))
        
        sparse_map.append((i,j,compute_continuity(indices_source[k], indices_embedded[k])))

    with open('continuity_sparse_map.npy', "wb") as f:
        np.save(f, sparse_map)
    
    proj_errors = _generate_interpolated_image_(sparse_map, resolution, method='nearest').T
    
    return proj_errors

def get_trustworthiness_errors(X_train_2d, X_train, resolution = 256):
    proj_errors = np.zeros((resolution, resolution))

    K = 10
    metric = "euclidean"

    X2d = X_train_2d.reshape((X_train_2d.shape[0], -1))
    Xnd = X_train.reshape((X_train.shape[0], -1))

    tree = KDTree(X2d, metric=metric)
    print("Created KDTree for X2d")
    indices_embedded = tree.query(X2d, k=K, return_distance=False)
    # Drop the actual point itself
    indices_embedded = indices_embedded[:, 1:]
    print("Indices embedded computed")
    tree = KDTree(Xnd, metric=metric)
    print("Created KDTree for Xnd")
    indices_source = tree.query(Xnd, k=Xnd.shape[0], return_distance=False)
    # Drop the actual point itself
    indices_source = indices_source[:, 1:]
    print("Indices source computed")   
    print(indices_source.shape)

    sparse_map = []

    for k in range(X2d.shape[0]):
        i, j = X2d[k]
        i, j = int(i * (resolution - 1)) , int(j * (resolution - 1))
        
        sparse_map.append((i,j,compute_trustworthiness(indices_source[k], indices_embedded[k])))

    with open('trustworthiness_sparse_map.npy', "wb") as f:
        np.save(f, sparse_map)
    
    proj_errors = _generate_interpolated_image_(sparse_map, resolution, method='nearest').T
    
    return proj_errors

def _generate_interpolated_image_(sparse_map, resolution, method='nearest'):
        X, Y, Z = [], [], []
        for (x, y, z) in sparse_map:
            X.append(x)
            Y.append(y)
            Z.append(z)
        X, Y, Z = np.array(X), np.array(Y), np.array(Z)
        xi = np.linspace(0, resolution-1, resolution)
        yi = np.linspace(0, resolution-1, resolution)
        return interpolate.griddata((X, Y), Z, (xi[None,:], yi[:,None]), method=method)

def get_points(X_train_2d, X_train, resolution = 256):
    X_train_2d = X_train_2d.reshape((X_train_2d.shape[0], -1))
    X_train = X_train.reshape((X_train.shape[0], -1))
    
    space2d = [(i / resolution, j / resolution) for i in range(resolution) for j in range(resolution)]
    spaceNd = np.zeros((resolution * resolution, X_train.shape[1]))
    
    tree = KDTree(X_train_2d, metric="euclidean")
    indices = tree.query(space2d, k=10, return_distance=False)
    indices = indices[:, 1:]
    
    for i in range(indices.shape[0]):
        indices_2d = indices[i]
        neighbors = X_train[indices_2d]
        spaceNd[i] = np.mean(neighbors, axis=0)
    
    return spaceNd, space2d


def save_workspace(inv_proj_errors, X_train, Y_train, X_test, Y_test, X_train_2d, X_test_2d, img, img_confidence, decoded):
    with open("test_tmp/inv_proj_errors.npy", "wb") as f:
        np.save(f, inv_proj_errors)
    with open("test_tmp/X_train.npy", "wb") as f:
        np.save(f, X_train)
    with open("test_tmp/Y_train.npy", "wb") as f:
        np.save(f, Y_train)
    with open("test_tmp/X_test.npy", "wb") as f:
        np.save(f, X_test)
    with open("test_tmp/Y_test.npy", "wb") as f:
        np.save(f, Y_test)
    with open("test_tmp/X_train_2d.npy", "wb") as f:
        np.save(f, X_train_2d)
    with open("test_tmp/X_test_2d.npy", "wb") as f:
        np.save(f, X_test_2d)
    with open("test_tmp/img.npy", "wb") as f:
        np.save(f, img)
    with open("test_tmp/img_confidence.npy", "wb") as f:
        np.save(f, img_confidence)
    with open("test_tmp/decoded.npy", "wb") as f:
        np.save(f, decoded)
    
def load_workspace():
    with open("test_tmp/inv_proj_errors.npy", "rb") as f:
        inv_proj_errors = np.load(f)
    with open("test_tmp/X_train.npy", "rb") as f:
        X_train= np.load(f)
    with open("test_tmp/Y_train.npy", "rb") as f:
        Y_train= np.load(f)
    with open("test_tmp/X_test.npy", "rb") as f:
        X_test= np.load(f)
    with open("test_tmp/Y_test.npy", "rb") as f:
        Y_test = np.load(f)
    with open("test_tmp/X_train_2d.npy", "rb") as f:
        X_train_2d = np.load(f)
    with open("test_tmp/X_test_2d.npy", "rb") as f:
        X_test_2d = np.load(f)
    with open("test_tmp/img.npy", "rb") as f:
        img = np.load(f)
    with open("test_tmp/img_confidence.npy", "rb") as f:
        img_confidence = np.load(f)
    with open("test_tmp/decoded.npy", "rb") as f:
        decoded = np.load(f)
    
    return inv_proj_errors, X_train, Y_train, X_test, Y_test, X_train_2d, X_test_2d, img, img_confidence, decoded


from numba import njit, prange
from umap.distances import named_distances

@njit(parallel=True)
def generate_indices(X, metric):
    n_samples = X.shape[0]
    dist_vector = np.zeros(n_samples, dtype=np.float64)
    indices = np.zeros((n_samples, n_samples), dtype=np.int64)
    
    for i in prange(n_samples):
        for j in prange(n_samples):
            dist_vector[j] = metric(X[i], X[j])

        indices[i] = np.argsort(dist_vector)
    
    return indices

def main():
    resolution = 256
    
    if not os.path.exists("test_tmp"):
        os.makedirs("test_tmp")
    
    X_train, Y_train, X_test, Y_test, X_train_2d, X_test_2d = load_data()
    
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    X = np.concatenate((X_train, X_test), axis=0)
    
    print("Strating KDTree query test")
    """
    start = time.time()
    tree = KDTree(X, metric="euclidean")
    end = time.time()
    print("KDTree building time: ", end - start)
    indx = tree.query(X, k=X.shape[0], return_distance=False)
    end_2 = time.time()
    print("KDTree query time: ", end_2 - end)
    # ~ 35 - 36 seconds
    
    leaf_size = 5000
    
    start = time.time()
    tree = KDTree(X, metric="euclidean", leaf_size=leaf_size)
    end = time.time()
    print("KDTree building time with leaf size ", leaf_size, end - start)
    indx = tree.query(X, k=X.shape[0], return_distance=False)
    end_2 = time.time()
    print("KDTree query time with leaf size ", leaf_size, end_2 - end)
    # ~ 29 - 30 seconds
    """
    
    start = time.time()
    indices = generate_indices(X, named_distances["euclidean"])
    end = time.time()
    print("Numba KDTree query time: ", end - start)
    print(indices)
    # ~ 9 seconds
    
    
    """   
    invNN = load_model(X_train, Y_train, X_test, Y_test, X_train_2d, X_test_2d)
    img, img_confidence, decoded = get_dbm_img(invNN, X_train_2d, X_test_2d, resolution = resolution)
    
    inv_proj_errors = get_inverse_projections_errors(decoded)
    
    save_workspace(inv_proj_errors, X_train, Y_train, X_test, Y_test, X_train_2d, X_test_2d, img, img_confidence, decoded)
    """   
    
    #inv_proj_errors, X_train, Y_train, X_test, Y_test, X_train_2d, X_test_2d, img, img_confidence, decoded = load_workspace()

    """
    proj_errors = get_continuity_errors(X_train_2d, X_train)
    
    with open("test_tmp/continuity_errors.npy", "wb") as f:
        np.save(f, proj_errors)
    
    proj_errors = get_trustworthiness_errors(X_train_2d, X_train)
    with open("test_tmp/trustworthiness_errors.npy", "wb") as f:
        np.save(f, proj_errors)
    """

    """
    with open("test_tmp/continuity_errors.npy", "rb") as f:
        continuity_errors = np.load(f)
    with open("test_tmp/trustworthiness_errors.npy", "rb") as f:
        trustworthiness_errors = np.load(f)
    
    
    proj_errors = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            proj_errors[i,j] = (continuity_errors[i,j] + trustworthiness_errors[i,j]) / 2
            #proj_errors[i,j] = continuity_errors[i,j]
            #proj_errors[i,j] = trustworthiness_errors[i, j]
    

    min_proj_error = np.min(proj_errors)
    max_proj_error = np.max(proj_errors)
    
    proj_errors = (proj_errors - min_proj_error) / (max_proj_error - min_proj_error)
    X_train_2d *= resolution - 1
    X_test_2d *= resolution - 1
    X_train_2d = X_train_2d.astype(int)
    X_test_2d = X_test_2d.astype(int)

    """
main()