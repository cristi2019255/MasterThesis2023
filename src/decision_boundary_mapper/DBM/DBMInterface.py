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
from math import ceil, floor, sqrt
from queue import PriorityQueue

from .tools import get_decode_pixel_priority, get_inv_proj_error

from ..utils import track_time_wrapper
from ..Logger import Logger, LoggerInterface

DBM_DEFAULT_RESOLUTION = 256

class DBMInterface:
    """ Decision Boundary Mapper Interface
    
    Methods to be implemented by the class that implements the DBMInterface:
        fit (X_train, Y_train, X_test, Y_test, epochs=10, batch_size=128, load_folder = None)
        generate_boundary_map (X_train, Y_train, X_test, Y_test, train_epochs=10, train_batch_size=128, resolution=DBM_DEFAULT_RESOLUTION)
        _predict2dspace_ (X)
    
    Example of usage: 
        class SuperBoundaryMapper(DBMInterface):
            def fit (....):
                # To be implemented
            def generate_boundary_map (....):
                # To be implemented
            def _predict2dspace_ (....):
                # To be implemented
    """    
    
    def __init__(self, classifier, logger:LoggerInterface=None):
        """Initializes the classifier and the logger

        Args:
            classifier (Any): The classifier to be used
            logger (LoggerInterface, optional): The logger for the outputting info messages. Defaults to None.
        """
        if logger is None:
            self.console = Logger(name="Decision Boundary Mapper - DBM")
        else:
            self.console = logger
        
        self.console.log("Loaded classifier: " + classifier.name + "")
        self.classifier = classifier
        
    def fit(self, X_train:np.ndarray, Y_train:np.ndarray, 
            X_test:np.ndarray, Y_test:np.ndarray,
            epochs:int=10, batch_size:int=128, load_folder:bool = None):
        """ Trains the classifier on the given data set.

        Args:
            X_train (np.ndarray): Training data set
            Y_train (np.ndarray): Training data labels
            X_test (np.ndarray): Testing data set
            Y_test (np.ndarray): Testing data labels
            epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            batch_size (int, optional): Train batch size. Defaults to 128.
        """
        pass
    
    def generate_boundary_map(self, 
                              X_train:np.ndarray, Y_train:np.ndarray, 
                              X_test:np.ndarray, Y_test:np.ndarray,
                              train_epochs:int=10, train_batch_size:int=128,
                              resolution:int=DBM_DEFAULT_RESOLUTION,
                              use_fast_decoding:bool=False,
                              projection:str=None
                              ):
        """ Generates a 2D boundary map of the classifier's decision boundary.

        Args:
            X_train (np.ndarray): Training data set
            Y_train (np.ndarray): Training data labels
            X_test (np.ndarray): Testing data set
            Y_test (np.ndarray): Testing data labels
            train_epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            train_batch_size (int, optional): Train batch size. Defaults to 128.
            show_predictions (bool, optional): If set to true 10 prediction examples are shown. Defaults to True.
            resolution (int, optional): _description_. Defaults to DBM_DEFAULT_RESOLUTION = 256.
            use_fast_decoding (bool, optional): If set to true the fast decoding algorithm is used. Defaults to False.
            projection (str, optional): The projection to be used for the 2D space. Defaults to None.
        Returns:
            np.array: A 2D numpy array with the decision boundary map
        
        """
        pass
    
    def _predict2dspace_(self, X2d: np.ndarray):
        """ Predicts the labels for the given 2D data set.

        Args:
            X2d (np.ndarray): The 2D data set
        
        Returns:
            predicted_labels (np.array): The predicted labels for the given 2D data set
            predicted_confidences (np.array): The predicted probabilities for the given 2D data set
            spaceNd (np.array): The decoded nD space
        """
        pass
    
    def generate_inverse_projection_errors(self, Xnd: np.ndarray = None):
        """ Calculates the inverse projection errors of the given data.

        Args:
            Xnd (np.array): The nd inverse projection of the data.
        
        Returns:
            errors (np.ndarray): The inverse projection errors matrix of the given data. (resolution x resolution)
        """
        if Xnd is None:
            if self.spaceNd is None:
                self.console.error("The nD data is not set, try to call the method 'generate_boundary_map' first.")
                raise Exception("The nD data is not set, try to call the method 'generate_boundary_map' first.")
            Xnd = self.spaceNd
        
        resolution = int(sqrt(len(Xnd)))
        assert resolution * resolution == len(Xnd)
        Xnd = Xnd.reshape((resolution,resolution,-1))
            
        self.console.log("Calculating the inverse projection errors of the given data")
        errors = np.zeros(Xnd.shape[:2])
        for i in range(Xnd.shape[0]):
            for j in range(Xnd.shape[1]):
                errors[i,j] = get_inv_proj_error(i,j, Xnd)
                
        # normalizing the errors to be in the range [0,1]
        errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
        return errors
    
    @track_time_wrapper
    def _get_img_dbm_(self, boundaries:tuple, resolution:int):
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
    def _get_img_dbm_fast_(self, boundaries:tuple, resolution:int, computational_budget=None):
        """
        This function generates the 2D image of the boundary map. It uses a fast algorithm to generate the image.
        
        ATTENTION: Highly recommended to use resolution = 2 ^ n, where n is an integer number (powers of 2).
        
        Args:
            boundaries (tuple): The boundaries of the 2D space (min_x, max_x, min_y, max_y)
            resolution (int): the resolution of the 2D image to be generated
            computational_budget (int, optional): The computational budget to be used. Defaults to None.
        Returns:
            img, img_confidence: The 2D image of the boundary map and a image with the confidence for each pixel
            space2d, spaceNd: The 2D space points and the nD space points
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
        
        predicted_labels, predicted_confidence, predicted_spaceNd = self._predict2dspace_(sparse_space)
        predicted_spaceNd = predicted_spaceNd.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION, -1))
        img_space_nD = np.zeros((resolution, resolution, predicted_spaceNd.shape[-1]))
        
        computational_budget -= INITIAL_RESOLUTION * INITIAL_RESOLUTION
        
        predictions = predicted_labels.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        confidences = predicted_confidence.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        spaceNd = predicted_spaceNd.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION, -1))
        
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x0, x1, y0, y1 = i * window_size, (i+1) * window_size, j * window_size, (j+1) * window_size
                img[x0:x1+ 1, y0:y1 + 1] = predictions[i,j]
                img_confidence[x0:x1 + 1, y0:y1+ 1] = confidences[i,j]
                img_space_nD[x0:x1 + 1, y0:y1 + 1] = spaceNd[i,j]
        # -------------------------------------
        
        # analyze the initial points and generate the priority queue        
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x, y = i * window_size + window_size / 2 - 0.5, j * window_size + window_size / 2 - 0.5
                priority = get_decode_pixel_priority(img, x, y, window_size, predictions[i,j])
                if priority != -1:
                    priority_queue.put((priority, (window_size, x, y)))

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
            
            computational_budget -= len(space)
            
            # decode the space
            predicted_labels, predicted_confidence, predicted_spaceNd = self._predict2dspace_(space)
            predicted_spaceNd = predicted_spaceNd.reshape((len(space), -1))
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
                spaceNd = predicted_spaceNd[index*NEIGHBORS_NUMBER:(index+1)*NEIGHBORS_NUMBER]
                
                new_window_size = window_size / NEIGHBORS_NUMBER
                
                for (x, y), label, conf, pointNd in zip(neighbors, neighbors_labels, confidences, spaceNd):
                    # all the pixels in the window are set to the same label
                    # starting from the top left corner of the window
                    # ending at the bottom right corner of the window
                    # the +1 is because the range function is not inclusive
                    if new_window_size >= 1:
                        x0, x1, y0, y1 = ceil(x-new_window_size), floor(x + new_window_size), ceil(y - new_window_size), floor(y + new_window_size)
                        new_img[x0:x1 + 1, y0:y1 + 1] = label
                        img_confidence[x0:x1 + 1, y0:y1 + 1] = conf
                        img_space_nD[x0:x1 + 1, y0:y1 + 1] = pointNd
                    else:
                        new_img[int(x), int(y)] = label
                        img_confidence[int(x), int(y)] = conf
                        img_space_nD[int(x), int(y)] = pointNd
                
                # update the priority queue after the window has been filled
                for (x, y), label in zip(neighbors, neighbors_labels):
                    priority = get_decode_pixel_priority(img, x, y, 2 * new_window_size, label)
                    if priority != -1:
                        priority_queue.put((priority, (2 * new_window_size, x, y)))
            
            # update the image        
            img = new_img
        
        img_space_2D = np.array([(i / resolution * (max_x - min_x) + min_x, j / resolution * (max_y - min_y) + min_y) for i in range(resolution) for j in range(resolution)])
        img_space_nD = img_space_nD.reshape((resolution * resolution, -1)) # flatten the array
        return img, img_confidence, img_space_2D, img_space_nD
