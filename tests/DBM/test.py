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

import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils import import_mnist_dataset, track_time_wrapper

COLORS_MAPPER = {
    0: [1,0,0], 
    1: [0,1,0], 
    2: [0,0,1], 
    3: [1,1,0], 
    4: [0,1,1], 
    5: [1,0,1], 
    6: [0.5,0.5,0.5], 
    7: [0.5,0,0], 
    8: [0,0.5,0], 
    9: [0,0,0.5]
}

def test_dbm(name="DBM"):
    path_img_fast = os.path.join(os.getcwd(), "models", name, "boundary_map_fast.npy")
    path_img_real = os.path.join(os.getcwd(), "models", name, "boundary_map.npy")

    path_confidence_fast = os.path.join(os.getcwd(), "models", name, "boundary_map_confidence_fast.npy")
    path_confidence_real = os.path.join(os.getcwd(), "models", name, "boundary_map_confidence.npy")

    with open(path_img_fast, "rb") as f:
        img1 = np.load(f)

    with open(path_img_real, "rb") as f:
        img2 = np.load(f)

    errors = 0
    for (i, j), z in np.ndenumerate(img1):
        if img2[i,j] != z:
            errors += 1
            print(f"Error at {i}, {j} : {img2[i,j]} != {z}")

    print(f"Percentage of errors: {errors / (img1.shape[0] * img1.shape[1]) * 100} %")     
    print(f"Number of errors: {errors}")

    with open(path_confidence_fast, "rb") as f:
        img1_conf = np.load(f)

    with open(path_confidence_real, "rb") as f:
        img2_conf = np.load(f)

    immg1 = np.zeros((img1.shape[0], img1.shape[1], 4))
    immg2 = np.zeros((img2.shape[0], img2.shape[1], 4))

    for (i, j), z in np.ndenumerate(img1):
        immg1[i,j] = COLORS_MAPPER[z] + [img1_conf[i,j]]

    for (i, j), z in np.ndenumerate(img2):
        immg2[i,j] = COLORS_MAPPER[z] + [img2_conf[i,j]]

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(immg1)
    ax4.imshow(immg2)

    plt.show()

import numpy as np
from sklearn.neighbors import KDTree

def compute_trustworthiness(indices_source, indices_embedded):
    rank = 0
    for i in range(len(indices_source)):
        if indices_source[i] != indices_embedded[i]:
            rank += 1
             
    return rank / len(indices_source)


SAMPLES_LIMIT = 5000
np.random.seed(42)
@track_time_wrapper
def test_projection_errors():
    (X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
        
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train, Y_train = X_train[:int(0.7*SAMPLES_LIMIT)], Y_train[:int(0.7*SAMPLES_LIMIT)]
    X_test, Y_test = X_test[:int(0.3*SAMPLES_LIMIT)], Y_test[:int(0.3*SAMPLES_LIMIT)]
    
    source = X_train.reshape((X_train.shape[0], -1))
    print(source.shape)
    
    """
    X2d = PROJECTION_METHODS["UMAP"](X_train)  
    if not os.path.exists(os.path.join(os.getcwd(), "data", "UMAP")):
        os.makedirs(os.path.join(os.getcwd(), "data", "UMAP"))    
    file_path = os.path.join(os.getcwd(), "data", "UMAP", "umap_embedding.npy")
    with open(file_path, 'wb') as f:
        np.save(f, X2d)
    """
    
    with open(os.path.join(os.getcwd(), "data", "UMAP", "umap_embedding.npy"), "rb") as f:
        embedding = np.load(f)
    
    K = 10
    metric = "euclidean"
    tree = KDTree(embedding, metric=metric)
    indices_embedded = tree.query(embedding, k=K, return_distance=False)
    # Drop the actual point itself
    indices_embedded = indices_embedded[:, 1:]
    print(indices_embedded.shape)
    
    tree = KDTree(source, metric=metric)
    indices_source = tree.query(source, k=K, return_distance=False)
    # Drop the actual point itself
    indices_source = indices_source[:, 1:]
    print(indices_source.shape)
   
    min_x, max_x = np.min(embedding[:,0]), np.max(embedding[:,0])
    min_y, max_y = np.min(embedding[:,1]), np.max(embedding[:,1])
    img = np.zeros((256,256))
    
    for ii in range(len(embedding)):
        x, y = embedding[ii]
        T = compute_trustworthiness(indices_source[ii], indices_embedded[ii])
        i, j = int((x - min_x) / (max_x - min_x) * 255), int((y - min_y) / (max_y - min_y) * 255)
        img[i,j] = T
     
    img *= 255
    plt.imshow(img, cmap="hot")
    plt.show()
