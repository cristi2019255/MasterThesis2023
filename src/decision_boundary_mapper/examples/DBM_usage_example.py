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
import os

from ..DBM import DBM
from ..GUI import DBMPlotterGUI

from .utils import *

def DBM_usage_example():
    # import the dataset
    X_train, X_test, Y_train, Y_test = import_data()
    
    # import the classifier
    classifier = import_classifier()
    
    # create the DBM
    dbm = DBM(classifier=classifier)
    
    # use the DBM to get the decision boundary map, if you don't have the 2D projection of the data
    # the DBM will get it for you, you just need to specify the projection method you would like to use (t-SNE, PCA or UMAP)
    img, img_confidence, _, _, _ = dbm.generate_boundary_map(X_train, Y_train, 
                                                                X_test, Y_test, 
                                                                resolution=256, 
                                                                load_folder=os.path.join("tmp", "MNIST", "DBM"),                                                                                                                             
                                                                projection="t-SNE")
    
    # if you have the 2D projection of the data, you can use the following function to get the decision boundary map
    """
    X2d_train, X2d_test = None, None # get the 2D projection of the data by yourself
    img, img_confidence, _, _, _ = dbm.generate_boundary_map(X_train, Y_train, 
                                                                X_test, Y_test,
                                                                X2d_train=X2d_train, 
                                                                X2d_test=X2d_test,
                                                                load_folder=os.path.join("tmp", "MNIST", "DBM"), 
                                                                use_fast_decoding=True,  
                                                                resolution=256)
    """                                                                  
    
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
    
    
    # use the dbm to get the inverse projection errors
    img_inverse_projection_errors = dbm.generate_inverse_projection_errors()
    # plot the inverse projection errors
    fig, ax = plt.subplots()
    ax.set_title("Inverse projection errors")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_ax = ax.imshow(img_inverse_projection_errors)
    fig.colorbar(img_ax, ax=ax)
    plt.show()
                                                           
    # use the dbm to get the projection errors
    img_projection_errors = dbm.generate_projection_errors()
    # plot the projection errors
    fig, ax = plt.subplots()
    ax.set_title("Projection errors")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_ax = ax.imshow(img_projection_errors)
    fig.colorbar(img_ax, ax=ax)
    plt.show()
    
def DBM_usage_example_GUI():
    # import the dataset
    X_train, X_test, Y_train, Y_test = import_data()
    
    # import the classifier
    classifier = import_classifier()
    
    # import the 2D projection of the data
    X2d_train, X2d_test = import_2d_data()
    
    # create the DBM
    dbm = DBM(classifier=classifier)
    
    # use the DBM to get the decision boundary map, if you don't have the 2D projection of the data
    # the DBM will get it for you, you just need to specify the projection method you would like to use (t-SNE, PCA or UMAP)
    dbm_info = dbm.generate_boundary_map(X_train, Y_train, 
                                         X_test, Y_test, 
                                         X2d_train=X2d_train,
                                         X2d_test=X2d_test,
                                         resolution=256, 
                                         load_folder=os.path.join("tmp", "MNIST", "DBM"),                                                                                                                             
                                         projection="t-SNE")
    
    img, img_confidence, encoded_training_data, encoded_testing_data, training_history  = dbm_info
    
    dbm_plotter_gui = DBMPlotterGUI(dbm_model = dbm,
                                    img = img,
                                    img_confidence = img_confidence,
                                    encoded_train = encoded_training_data, 
                                    encoded_test = encoded_testing_data,
                                    X_train = X_train,
                                    Y_train = Y_train,
                                    X_test = X_test,
                                    Y_test = Y_test,                                    
                                    save_folder=os.path.join("tmp", "MNIST", "DBM"), # this is the folder where the DBM will save the changes in data the user makes
                                    projection_technique="t-SNE",
                                    )
    dbm_plotter_gui.start()