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

import numpy as np
import matplotlib.pyplot as plt

from ..DBM import SDBM
from ..GUI import DBMPlotterGUI

from .utils import *

def SDBM_usage_example():
    # import the dataset
    X_train, X_test, Y_train, Y_test = import_data()
    
    # import the classifier
    classifier = import_classifier()
    
    # create the DBM
    sdbm = SDBM(classifier=classifier)
    
    # use the SDBM to get the decision boundary map
    img, img_confidence, _, _, _ = sdbm.generate_boundary_map(X_train, Y_train, 
                                                                X_test, Y_test,
                                                                load_folder=os.path.join("tmp", "MNIST", "SDBM"),    
                                                                resolution=256)
    
    
    # make the decision boundary map pretty, by adding the colors and the confidence
    COLORS_MAPPER = {
        -2: [0,0,0], # setting original test data to black
        -1: [1,1,1], # setting original train data to white
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

    color_img = np.zeros((img.shape[0], img.shape[1], 4))
    for i, j in np.ndindex(img.shape):       
        color_img[i,j] = COLORS_MAPPER[img[i,j]] + [img_confidence[i,j]]
    
    # plot the decision boundary map
    plt.title("Decision boundary map")
    plt.axis("off")
    plt.imshow(color_img)
    plt.show()
    
    # use the SDBM to get the inverse projection errors
    img_inverse_projection_errors = sdbm.generate_inverse_projection_errors(resolution=256)
    # plot the inverse projection errors
    fig, ax = plt.subplots()
    ax.set_title("Inverse projection errors")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_ax = ax.imshow(img_inverse_projection_errors)
    fig.colorbar(img_ax, ax=ax)
    plt.show()
    
    # use the SDBM to get the projection errors
    img_projection_errors = sdbm.generate_projection_errors()
    # plot the projection errors
    fig, ax = plt.subplots()
    ax.set_title("Projection errors")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_ax = ax.imshow(img_projection_errors)
    fig.colorbar(img_ax, ax=ax)
    plt.show()
    
def SDBM_usage_example_GUI():
    # import the dataset
    X_train, X_test, Y_train, Y_test = import_data()
    
    Y = np.copy(Y_train)
    
    # use the opf to generate the labels for the training data if the data is not completely labelled
    Y_train = opf(X_train=X_train, Y_train=Y_train)
    
    # generate a classifier
    classifier = generate_classifier(X_train, Y_train)
    
    # import the classifier
    #classifier = import_classifier()
    
    # create the DBM
    sdbm = SDBM(classifier=classifier)
    
    
    # use the DBM to get the decision boundary map, if you don't have the 2D projection of the data
    # the DBM will get it for you, you just need to specify the projection method you would like to use (t-SNE, PCA or UMAP)
    dbm_info = sdbm.generate_boundary_map(X_train, Y, 
                                         X_test, Y_test,
                                         resolution=256,
                                         load_folder=os.path.join("tmp", "MNIST", "SDBM"),                                                                                                                             
                                         )
    
    img, img_confidence, encoded_training_data, encoded_testing_data, training_history  = dbm_info
    
    dbm_plotter_gui = DBMPlotterGUI(dbm_model = sdbm,
                                    img = img,
                                    img_confidence = img_confidence,
                                    encoded_train = encoded_training_data, 
                                    encoded_test = encoded_testing_data,
                                    X_train = X_train,
                                    Y_train = Y_train,
                                    X_test = X_test,
                                    Y_test = Y_test,
                                    save_folder=os.path.join("tmp", "MNIST", "SDBM"), # this is the folder where the DBM will save the changes in data the user makes                                    
                                    )
    dbm_plotter_gui.start()