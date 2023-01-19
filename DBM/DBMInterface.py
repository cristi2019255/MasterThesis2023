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

from utils.Logger import Logger
import numpy as np
from DBM.tools import get_decode_pixel_priority
from math import ceil, floor
from queue import PriorityQueue
from utils.tools import track_time_wrapper

DBM_DEFAULT_RESOLUTION = 256


class DBMInterface:
    
    def __init__(self, classifier, logger=None):
        if logger is None:
            self.console = Logger(name="Decision Boundary Mapper - DBM")
        else:
            self.console = logger
        
        self.console.log("Loaded classifier: " + classifier.name + "")
        self.classifier = classifier
        
    def fit(self, X_train, Y_train, X_test, Y_test, epochs=10, batch_size=128, load_folder = None):
        """ Trains the classifier on the given data set.

        Args:
            X_train (np.array): Training data set
            Y_train (np.array): Training data labels
            X_test (np.array): Testing data set
            Y_test (np.array): Testing data labels
            epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            batch_size (int, optional): Train batch size. Defaults to 128.
        """
        pass
    
    def generate_boundary_map(self, 
                              X_train, Y_train, X_test, Y_test,
                              train_epochs=10, 
                              train_batch_size=128,
                              resolution=DBM_DEFAULT_RESOLUTION):
        """ Generates a 2D boundary map of the classifier's decision boundary.

        Args:
            X_train (np.array): Training data set
            Y_train (np.array): Training data labels
            X_test (np.array): Testing data set
            Y_test (np.array): Testing data labels
            train_epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            train_batch_size (int, optional): Train batch size. Defaults to 128.
            show_predictions (bool, optional): If set to true 10 prediction examples are shown. Defaults to True.
            resolution (_type_, optional): _description_. Defaults to DBM_DEFAULT_RESOLUTION.
        
        Returns:
            np.array: A 2D numpy array with the decision boundary map
        
        """
        pass
    
    
    def _predict2dspace_(self, X2d):
        """ Predicts the labels for the given 2D data set.

        Args:
            X2d (np.array): The 2D data set
        
        Returns:
            np.array: The predicted labels for the given 2D data set
            np.array: The predicted probabilities for the given 2D data set
        """
        pass
    
    @track_time_wrapper
    def _get_img_dbm_(self, boundaries, resolution):
        """ This function generates the 2D image of the boundary map using the trained autoencoder and classifier.

        Args:
            boundaries (tuple): The boundaries of the 2D space (min_x, max_x, min_y, max_y)
            resolution (int): The resolution of the 2D image to be generated 

        Returns:
            img (np.array): The 2D image of the boundary map
            img_confidence (np.array): The confidence of map of each pixel of the 2D image of the boundary map
            space2d (np.array): The 2D space points
            spaceNd (np.array): The nD space points
      
        Example:
            >>> img, img_confidence, space2d, spaceNd = self._get_img_dbm_((0, 1, 0, 1), resolution = 100)
        """
        min_x, max_x, min_y, max_y = boundaries
        space2d = np.array([(i / resolution * (max_x - min_x) + min_x, j / resolution * (max_y - min_y) + min_y) for i in range(resolution) for j in range(resolution)])
        self.console.log("Predicting labels for the 2D boundary mapping using the nD data and the trained classifier...")
        predicted_labels, predicted_confidence, spaceNd = self._predict2dspace_(space2d)
        img = predicted_labels.reshape((resolution, resolution))
        img_confidence = predicted_confidence.reshape((resolution, resolution))
        return (img, img_confidence, space2d, spaceNd)
    
    @track_time_wrapper
    def _get_img_dbm_fast_(self, boundaries, resolution, computational_budget=None):
        """
        This function generates the 2D image of the boundary map. It uses a fast algorithm to generate the image.
        
        ATTENTION: Highly recommended to use resolution = 2 ^ n, where n is an integer number (powers of 2).
        
        Args:
            boundaries (tuple): The boundaries of the 2D space (min_x, max_x, min_y, max_y)
            resolution (int): the resolution of the 2D image to be generated

        Returns:
            img, img_confidence: The 2D image of the boundary map and a image with the confidence for each pixel
        
        Example:
            >>> img, img_confidence = self._get_img_dbm_fast_(boundaries=(0, 1, 0, 1), resolution=32, computational_budget=1000)
        """
        # ------------------------------------------------------------
        # Setting the initial parameters
        INITIAL_RESOLUTION = 32
        NEIGHBORS_NUMBER = 4
        
        if computational_budget is None:
            computational_budget = resolution * resolution
        
        if (resolution % INITIAL_RESOLUTION != 0):
            self.console.warn(f"The required resolution is not a multiple of the initial resolution ({INITIAL_RESOLUTION} x {INITIAL_RESOLUTION})")
            self.console.log("The resolution will be set to the closest multiple of the initial resolution")
            resolution = int(resolution / INITIAL_RESOLUTION) * INITIAL_RESOLUTION   
        
        window_size = int(resolution / INITIAL_RESOLUTION)
        
        min_x, max_x, min_y, max_y = boundaries
        # ------------------------------------------------------------
        
        # ------------------------------------------------------------
        img = np.zeros((resolution, resolution))
        img_confidence = np.zeros((resolution, resolution))
        priority_queue = PriorityQueue()
        # ------------------------------------------------------------
                
        # ------------------------------------------------------------
        # generate the initial points
        sparse_space = []
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x = (i * window_size + window_size / 2) / resolution * (max_x - min_x) + min_x
                y = (j * window_size + window_size / 2) / resolution * (max_y - min_y) + min_y                
                sparse_space.append((x, y))
        
        predicted_labels, predicted_confidence, _ = self._predict2dspace_(sparse_space)
        
        computational_budget -= INITIAL_RESOLUTION * INITIAL_RESOLUTION
        
        predictions = predicted_labels.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        confidences = predicted_confidence.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                img[i*window_size:(i+1)*window_size + 1, j*window_size:(j+1)*window_size + 1] = predictions[i,j]
                img_confidence[i*window_size:(i+1)*window_size + 1, j*window_size:(j+1)*window_size + 1] = confidences[i,j]
        # -------------------------------------
        
        # analyze the initial points and generate the priority queue        
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x, y = i * window_size + window_size / 2 - 0.5, j * window_size + window_size / 2 - 0.5
                priority = get_decode_pixel_priority(img, x, y, window_size, predictions[i,j])
                if priority != -1:
                    item = (window_size, x, y)
                    priority_queue.put((priority, item))

        # -------------------------------------
        # start the iterative process of filling the image
        while computational_budget > 0 and not priority_queue.empty():    
            # take the highest priority task
            priority, item = priority_queue.get()
            
            # getting all the items with the same priority
            items = [item]
            if not priority_queue.empty():
                next_priority, next_item = priority_queue.get()
                while priority == next_priority:
                    items.append(next_item)
                    if priority_queue.empty():
                        break
                    next_priority, next_item = priority_queue.get()
                if priority != next_priority:
                    priority_queue.put((next_priority, next_item))
            
            
            space2d, indices = [], []
            single_points_space, single_points_indices = [], []
            valid_items = []
            
            for item in items:            
                (window_size, i, j) = item
                
                if window_size == 1:
                    single_points_indices += [(int(i), int(j))]
                    single_points_space += [(i / resolution * (max_x - min_x) + min_x, j / resolution * (max_y - min_y) + min_y)]
                    continue
                
                window_size = window_size / NEIGHBORS_NUMBER
                neighbors = [(i - window_size, j - window_size), (i - window_size, j + window_size), (i + window_size, j - window_size), (i + window_size, j + window_size)]
                space2d += [(x / resolution * (max_x - min_x) + min_x, y / resolution * (max_y - min_y) + min_y) for (x, y) in neighbors]
                indices += neighbors                                                
                valid_items.append(item)
            
            
            space = space2d + single_points_space
            
            # check if the computational budget is enough and update it
            if computational_budget - len(space) < 0:
                self.console.warn("Computational budget exceeded, stopping the process")
                break
            else:
                computational_budget -= len(space)
            
            # decode the space
            predicted_labels, predicted_confidence, _ = self._predict2dspace_(space)
            
            # copy the image to a new one, the new image will be updated with the new labels, the old one will be used to calculate the priorities
            new_img = np.copy(img)            
            
            # fill the new image with the single points
            single_points_labels = predicted_labels[len(space2d):]
            for i in range(len(single_points_indices)):
                new_img[single_points_indices[i]] = single_points_labels[i]
            
            predicted_labels = predicted_labels[:len(space2d)]
            
            # fill the new image with the new labels and update the priority queue
            for index in range(len(valid_items)):
                (window_size, i, j) = valid_items[index]
                neighbors = indices[index*NEIGHBORS_NUMBER:(index+1)*NEIGHBORS_NUMBER]
                neighbors_labels = predicted_labels[index*NEIGHBORS_NUMBER:(index+1)*NEIGHBORS_NUMBER]
                confidences = predicted_confidence[index*NEIGHBORS_NUMBER:(index+1)*NEIGHBORS_NUMBER]
                
                new_window_size = window_size / NEIGHBORS_NUMBER
                
                for (x, y), label, conf in zip(neighbors, neighbors_labels, confidences):
                    # all the pixels in the window are set to the same label
                    # starting from the top left corner of the window
                    # ending at the bottom right corner of the window
                    # the +1 is because the range function is not inclusive
                    if new_window_size >= 1:
                        new_img[ceil(x-new_window_size):floor(x + new_window_size) + 1, ceil(y - new_window_size):floor(y + new_window_size) + 1] = label
                        img_confidence[ceil(x-new_window_size):floor(x + new_window_size) + 1, ceil(y - new_window_size):floor(y + new_window_size) + 1] = conf
                    else:
                        new_img[int(x), int(y)] = label
                        img_confidence[int(x), int(y)] = conf
                
                # update the priority queue after the window has been filled
                for (x, y), label in zip(neighbors, neighbors_labels):
                    priority = get_decode_pixel_priority(img, x, y, 2 * new_window_size, label)
                    if priority != -1:
                        priority_queue.put((priority, (2 * new_window_size, x, y)))
            
            # update the image        
            img = new_img
                
        return img, img_confidence
