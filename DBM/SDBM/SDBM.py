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
from DBM.DBMInterface import DBMInterface
from DBM.DBMInterface import DBM_DEFAULT_RESOLUTION
from DBM.SDBM.Autoencoder import DEFAULT_MODEL_PATH, Autoencoder, build_autoencoder
from DBM.tools import get_decode_pixel_priority, get_inv_proj_error, get_proj_error
from queue import PriorityQueue
from math import ceil, floor
from utils.tools import track_time_wrapper
        
class SDBM(DBMInterface):
    """
        SDBM - Self Decision Boundary Mapper
    """
    def __init__(self, classifier, logger=None):
        super().__init__(classifier, logger)
        self.autoencoder = None

    def fit(self, 
            X_train, 
            Y_train, 
            X_test, 
            Y_test, 
            epochs=10,
            batch_size=128,
            load_folder = DEFAULT_MODEL_PATH):
        
        # Train an autoencoder on the training data (this will be used to reduce the dimensionality of the data) nD -> 2D
        try:
            autoencoder = Autoencoder(folder_path = load_folder, load = True)
            self.console.log("Loaded autoencoder from disk")
        except Exception as e:
            self.console.log("Could not load autoencoder from disk. Training a new one.")
            data_shape = X_train.shape[1:]
            autoencoder = build_autoencoder(self.classifier, data_shape, show_summary=True)
            autoencoder.fit(X_train, Y_train, X_test, Y_test, epochs=epochs, batch_size=batch_size)
        
        #if show_predictions:
        #    autoencoder.show_predictions(X_test[:20], Y_test[:20])

        self.autoencoder = autoencoder
        self.classifier = autoencoder.classifier
        return autoencoder
    
    def generate_boundary_map(self, 
                              X_train, Y_train, X_test, Y_test,
                              train_epochs=10, 
                              train_batch_size=128,
                              show_encoded_corpus=True,
                              resolution=DBM_DEFAULT_RESOLUTION, 
                              ):
        
        # making sure that the data is of the correct type
        assert type(X_train) == np.ndarray
        assert type(Y_train) == np.ndarray
        assert type(X_test) == np.ndarray
        assert type(Y_test) == np.ndarray
        
        
        # first train the autoencoder if it is not already trained
        if self.autoencoder is None:
            self.fit(X_train, 
                    Y_train, 
                    X_test, 
                    Y_test, 
                    epochs=train_epochs, 
                    batch_size=train_batch_size)

        # encoder the train and test data and show the encoded data in 2D space
        self.console.log("Encoding the training data to 2D space")
        encoded_training_data = self.autoencoder.encode(X_train)
        self.console.log("Encoding the testing data to 2D space")
        encoded_testing_data = self.autoencoder.encode(X_test)            
        
        # Skip the visualization of the encoded data
        #if show_encoded_corpus:
        #    plt.figure(figsize=(20, 20))
        #    plt.title("Encoded data in 2D space")
        #    plt.axis('off')
        #    plt.plot(encoded_training_data[:,0], encoded_training_data[:,1], 'ro', label="Training data", alpha=0.5)
        #    plt.plot(encoded_testing_data[:,0], encoded_testing_data[:,1], 'bo', label="Testing data", alpha=0.5)
        #    plt.show()
            
        # getting the max and min values for the encoded data
        min_x = min(np.min(encoded_training_data[:,0]), np.min(encoded_testing_data[:,0]))
        max_x = max(np.max(encoded_training_data[:,0]), np.max(encoded_testing_data[:,0]))
        min_y = min(np.min(encoded_training_data[:,1]), np.min(encoded_testing_data[:,1]))
        max_y = max(np.max(encoded_training_data[:,1]), np.max(encoded_testing_data[:,1]))
        
        # transform the encoded data to be in the range [0, resolution]
        encoded_training_data[:,0] = (encoded_training_data[:,0] - min_x) / (max_x - min_x) * (resolution - 1)
        encoded_testing_data[:,0] = (encoded_testing_data[:,0] - min_x) / (max_x - min_x) * (resolution - 1)
        encoded_training_data[:,1] = (encoded_training_data[:,1] - min_y) / (max_y - min_y) * (resolution - 1)
        encoded_testing_data[:,1] = (encoded_testing_data[:,1] - min_y) / (max_y - min_y) * (resolution - 1)
        
        encoded_training_data = encoded_training_data.astype(int)
        encoded_testing_data = encoded_testing_data.astype(int)
        
        # generate the 2D image in the encoded space
        
        img, img_confidence = self.get_img_dbm_fast(min_x, min_y, max_x, max_y, resolution)
        
        with open(os.path.join(DEFAULT_MODEL_PATH, "fast_boundary_map.npy"), 'wb') as f:
            np.save(f, img)
        with open(os.path.join(DEFAULT_MODEL_PATH, "fast_boundary_map_confidence.npy"), 'wb') as f:
            np.save(f, img_confidence)
        
        img, img_confidence, spaceNd, space2d, predicted_labels = self.get_img_dbm(min_x, min_y, max_x, max_y, resolution)
        
        save_img_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map")
        save_img_confidence_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map_confidence")
        with open(f"{save_img_path}.npy", 'wb') as f:
            np.save(f, img)
        with open(f"{save_img_confidence_path}.npy", 'wb') as f:
            np.save(f, img_confidence)
        
        for [i,j] in encoded_training_data:
            img[i,j] = -1
            img_confidence[i,j] = 1
        for [i,j] in encoded_testing_data:
            img[i,j] = -2
            img_confidence[i,j] = 1
        
        img_projection_errors = self.get_projection_errors(spaceNd, space2d, predicted_labels, resolution)
        img_inverse_projection_errors = self.get_inverse_projection_errors(spaceNd.reshape((resolution, resolution, -1)))
      
        return (img, img_confidence, img_projection_errors, img_inverse_projection_errors, encoded_training_data, encoded_testing_data)
    
    @track_time_wrapper
    def get_img_dbm(self, min_x, min_y, max_x, max_y, resolution):
        space2d = np.array([(i / resolution * (max_x - min_x) + min_x, j / resolution * (max_y - min_y) + min_y) for i in range(resolution) for j in range(resolution)])
        self.console.log("Decoding the 2D space... 2D -> nD")
        spaceNd = self.autoencoder.decode(space2d)
        self.console.log("Predicting labels for the 2D boundary mapping using the nD data and the trained classifier...")
        predictions = self.classifier.predict(spaceNd)
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        predicted_confidence = np.array([np.max(p) for p in predictions])
        img = predicted_labels.reshape((resolution, resolution))
        img_confidence = predicted_confidence.reshape((resolution, resolution))
        
        return (img, img_confidence, space2d, spaceNd, predicted_labels)
    
    @track_time_wrapper
    def get_img_dbm_fast(self, min_x, min_y, max_x, max_y, resolution):
        self.console.log("Decoding the 2D space... 2D -> nD")
        INITIAL_RESOLUTION = 16
        COMPUTATIONAL_BUDGET = 40000 # number of points that can be computed
        img = np.zeros((resolution, resolution)) - 1000 # -1000 is the value for the unknown points, this is used to see if the point was computed or not
        img_confidence = np.zeros((resolution, resolution))
        
        window_size = int(resolution / INITIAL_RESOLUTION)
                
        # generate the initial points
        sparse_space_2d = []
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                float_x = i * window_size + window_size / 2
                float_y = j * window_size + window_size / 2
                x = float_x / resolution * (max_x - min_x) + min_x
                y = float_y / resolution * (max_y - min_y) + min_y                
                sparse_space_2d.append((x, y))
        
        COMPUTATIONAL_BUDGET -= INITIAL_RESOLUTION * INITIAL_RESOLUTION
        
        spaceNd = self.autoencoder.decode(sparse_space_2d)
        predictions = self.classifier.predict(spaceNd, verbose=0)
        predictions_labels = np.array([np.argmax(p) for p in predictions])
        predictions_confidence = np.array([np.max(p) for p in predictions])
        
        img_pred = predictions_labels.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        confidences = predictions_confidence.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                img[i*window_size:(i+1)*window_size + 1, j*window_size:(j+1)*window_size + 1] = img_pred[i,j]
                img_confidence[i*window_size:(i+1)*window_size + 1, j*window_size:(j+1)*window_size + 1] = confidences[i,j]

        # -------------------------------------
        # analyze the initial points and generate the priority queue
        priority_queue = PriorityQueue()
                
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x, y = i * window_size + window_size / 2 - 0.5, j * window_size + window_size / 2 - 0.5
                priority = get_decode_pixel_priority(img, x, y, window_size, img_pred[i,j])
                if priority != -1:
                    item = (window_size, x, y)
                    priority_queue.put((priority, item))

        # -------------------------------------
        # start the iterative process of filling the image
        while not priority_queue.empty():    
            # take the highest priority task
            priority, item = priority_queue.get()
            
            # getting all the items with the same priority
            items = [item]
            if not priority_queue.empty():
                next_priority, next_item = priority_queue.get()
                while priority == next_priority:
                    items.append(next_item)
                    if not priority_queue.empty():
                        next_priority, next_item = priority_queue.get()
                    else:
                        break
                if priority != next_priority:
                    priority_queue.put((next_priority, next_item))
        
            
            space2d, indices = [], []
            valid_items = []
            single_points_space, single_points_indices = [], []
            SPLITTING_FACTOR = 2
            NEIGHBORS_NUMBER = SPLITTING_FACTOR * SPLITTING_FACTOR
            for item in items:            
                (window_size, i, j) = item
                
                if window_size == 1:
                    single_points_indices += [(int(i), int(j))]
                    single_points_space += [(i / resolution * (max_x - min_x) + min_x, j / resolution * (max_y - min_y) + min_y)]
                    continue
                
                window_size = window_size / NEIGHBORS_NUMBER
                
                neighbors = [(i - window_size, j - window_size), (i - window_size, j + window_size), 
                              (i + window_size, j - window_size), (i + window_size, j + window_size)]
                
                space_chunk = [(x / resolution * (max_x - min_x) + min_x, y / resolution * (max_y - min_y) + min_y) for (x, y) in neighbors]
                                                
                space2d += space_chunk
                indices += neighbors
                valid_items.append(item)
            
            """
            if COMPUTATIONAL_BUDGET - len(space2d) < 0:
                max_allowed = COMPUTATIONAL_BUDGET // NEIGHBORS_NUMBER
                items = items[:max_allowed]
                COMPUTATIONAL_BUDGET -= max_allowed * NEIGHBORS_NUMBER
                space2d = space2d[:max_allowed * NEIGHBORS_NUMBER]
                #print(f"Computational budget exceeded, only {max_allowed} items will be computed")                        
            else:
                COMPUTATIONAL_BUDGET -= len(space2d) + len(single_points_space)
            """
                
            # decode the space
            space = space2d + single_points_space
            spaceNd = self.autoencoder.decode(space)
            predictions = self.classifier.predict(spaceNd, verbose=0)
            labels = [np.argmax(p) for p in predictions]
            labels_confidence = [np.max(p) for p in predictions]
            
            new_img = np.copy(img)            
            # fill the image with the single points
            single_points_labels = labels[len(space2d):]
            for i in range(len(single_points_indices)):
                new_img[single_points_indices[i]] = single_points_labels[i]
            
            labels = labels[:len(space2d)]
            
            # fill the image with the new labels and update the priority queue
            for item in valid_items:
                (window_size, i, j) = item
                neighbors = indices[:NEIGHBORS_NUMBER]
                indices = indices[NEIGHBORS_NUMBER:]
                neighbors_labels = labels[:NEIGHBORS_NUMBER]
                labels = labels[NEIGHBORS_NUMBER:]
                confidences = labels_confidence[:NEIGHBORS_NUMBER]
                labels_confidence = labels_confidence[NEIGHBORS_NUMBER:]
                
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
                    priority = get_decode_pixel_priority(img, x, y, 2*new_window_size, label)
                    if priority != -1:
                        priority_queue.put((priority, (2*new_window_size, x, y)))
                    
            img = new_img
                
        return img, img_confidence
        
        
    def get_projection_errors(self, Xnd, X2d, labels, resolution):
        """ Calculates the projection errors of the given data.

        Args:
            Xnd (np.array): The data to be projected.
            X2d (np.array): The 2D projection of the data.
        Returns:
            np.array: The projection errors matrix of the 
        """
        self.console.log("Calculating the projection errors of the given data")
        errors = np.zeros(X2d.shape[0])
        
        distances_2d = np.array([np.linalg.norm(x) for x in X2d])
        distances_nd = np.array([np.linalg.norm(x) for x in Xnd])
        
        indices_2d = np.argsort(distances_2d)
        indices_nd = np.argsort(distances_nd)
        
        self.console.log(f"Indices 2d: {indices_2d[:10]}")
        self.console.log(f"Indices nd: {indices_nd[:10]}")
        
        for i in range(X2d.shape[0]):
            errors[i] = get_proj_error(i, indices_nd, indices_2d, labels)    
        
        # reshaping the errors to be in the shape of the 2d space
        errors = errors.reshape((resolution, resolution))
        return errors
    
    def get_inverse_projection_errors(self, Xnd):
        """ Calculates the inverse projection errors of the given data.

        Args:
            X2d (np.array): The 2d projection of the data.
            Xnd (np.array): The nd inverse projection of the data.
        """
        self.console.log("Calculating the inverse projection errors of the given data")
        errors = np.zeros(Xnd.shape[:2])
        for i in range(Xnd.shape[0]):
            for j in range(Xnd.shape[1]):
                errors[i,j] = get_inv_proj_error(i,j, Xnd)
                
        # normalizing the errors to be in the range [0,1]
        errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
        return errors