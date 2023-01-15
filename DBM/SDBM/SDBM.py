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
import time
import numpy as np
from DBM.DBMInterface import DBMInterface
from DBM.DBMInterface import DBM_DEFAULT_RESOLUTION
from DBM.SDBM.Autoencoder import DEFAULT_MODEL_PATH, Autoencoder, build_autoencoder
from DBM.tools import get_inv_proj_error, get_proj_error
from queue import PriorityQueue
from numba import jit

@jit
def get_priority(img, i, j, window_size, label):
    resolution = img.shape[0]
    # getting the 4 neighbors
    i, j = int(i), int(j)
    neighbors = []
    if i - window_size/2 > window_size:
        neighbors.append(img[i - window_size, j])
    if i + window_size/2 < resolution - window_size:
        neighbors.append(img[i + window_size + 1, j])
    if j - window_size/2 > window_size:
        neighbors.append(img[i + 1, j - window_size])
    if j + window_size/2 < resolution - window_size:
        neighbors.append(img[i, j + window_size + 1])
    
    if len(neighbors) == 0:
        print("No neighbors!!!!")
            
    cost = 0
    for neighbour_label in neighbors:
        if neighbour_label == label:
            cost += 1
    cost = cost / len(neighbors)
    cost = cost * window_size        
    if cost == 0:
        return -1
    return 1/cost
        

def track_time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds")
        return result
    return wrapper

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
        
        img = self.get_img_dbm_fast(min_x, min_y, max_x, max_y, resolution)
        
        self.console.log(f"Generated boundary map {img}")
        with open(os.path.join(DEFAULT_MODEL_PATH, "fast_boundary_map.npy"), 'wb') as f:
            np.save(f, img)
        
        img, img_confidence, spaceNd, space2d, predicted_labels = self.get_img_dbm(min_x, min_y, max_x, max_y, resolution)
        
        for [i,j] in encoded_training_data:
            img[i,j] = -1
            img_confidence[i,j] = 1
        for [i,j] in encoded_testing_data:
            img[i,j] = -2
            img_confidence[i,j] = 1
        
        save_img_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map")
        save_img_confidence_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map_confidence")
        with open(f"{save_img_path}.npy", 'wb') as f:
            np.save(f, img)
        with open(f"{save_img_confidence_path}.npy", 'wb') as f:
            np.save(f, img_confidence)
        
        img_projection_errors = self.get_projection_errors(spaceNd, space2d, predicted_labels, resolution)
        img_inverse_projection_errors = self.get_inverse_projection_errors(spaceNd.reshape((resolution, resolution, -1)))
      
        return (img, img_confidence, img_projection_errors, img_inverse_projection_errors, encoded_training_data, encoded_testing_data)
    
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
        INITIAL_RESOLUTION = 64
        COMPUTATIONAL_BUDGET = 20000 # number of points that can be computed
        img = np.zeros((resolution, resolution)) - 1000 # -1000 is the value for the unknown points, this is used to see if the point was computed or not
        window_size = int(resolution / INITIAL_RESOLUTION)
        
        sparse_space_2d = []
        
        # generate the initial points
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
        img_pred = predictions_labels.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                img[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size] = img_pred[i,j]
        
        priority_queue = PriorityQueue()
        
        
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x, y = i * window_size + window_size / 2, j * window_size + window_size / 2
                priority = get_priority(img, x, y, window_size, img_pred[i,j])
                item = (window_size, x, y)
                if priority != -1:
                    priority_queue.put((priority, item))

        iteration = 0
        while COMPUTATIONAL_BUDGET > 0 and not priority_queue.empty():
            # take the highest priority task
            priority, item = priority_queue.get()
            if priority == -1:
                continue
            
            print(f"Current priority: {priority}")
            
            items = [item]
            if not priority_queue.empty():        
                next_priority, next_item = priority_queue.get()
                while next_priority == priority:
                    items.append(next_item)
                    if priority_queue.empty():
                        break
                    next_priority, next_item = priority_queue.get()
                priority_queue.put((next_priority, next_item))
                
            space, indices, window_sizes = [], [], []
            print(f"Number of items with the same priority: {len(items)}")
            for item in items:            
                        (window_size, i, j) = item
                        window_size = int(window_size / 2)
                        i1, j1 = i - window_size, j - window_size
                        i2, j2 = i - window_size, j + window_size
                        i3, j3 = i + window_size, j - window_size
                        i4, j4 = i + window_size, j + window_size
                        
                        x1, y1 = i1 / resolution * (max_x - min_x) + min_x, j1 / resolution * (max_y - min_y) + min_y                
                        x2, y2 = i2 / resolution * (max_x - min_x) + min_x, j2 / resolution * (max_y - min_y) + min_y
                        x3, y3 = i3 / resolution * (max_x - min_x) + min_x, j3 / resolution * (max_y - min_y) + min_y
                        x4, y4 = i4 / resolution * (max_x - min_x) + min_x, j4 / resolution * (max_y - min_y) + min_y
                        
                        space += [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                        indices += [(i1, j1), (i2, j2), (i3, j3), (i4, j4)]
                        window_sizes += [window_size]
            
            COMPUTATIONAL_BUDGET -= 4 * len(items)
            if COMPUTATIONAL_BUDGET < 0:
                break
                    
            spaceNd = self.autoencoder.decode(space)
            preds = self.classifier.predict(spaceNd, verbose=0)
            
            preds = [np.argmax(p) for p in preds]
            
            for item in items:
                window_size = window_sizes.pop(0)
                if window_size >= 1:
                    i1, j1 = indices.pop(0)
                    i2, j2 = indices.pop(0)
                    i3, j3 = indices.pop(0)
                    i4, j4 = indices.pop(0)
                    w = window_size
                    label1 = preds.pop(0)
                    label2 = preds.pop(0)
                    label3 = preds.pop(0)
                    label4 = preds.pop(0)
                    x1, y1 = int(i1), int(j1)
                    x2, y2 = int(i2), int(j2)
                    x3, y3 = int(i3), int(j3)
                    x4, y4 = int(i4), int(j4)
                    
                    img[x1-w-1:x1+w, y1-w-1:y1+w] = label1
                    img[x2-w-1:x2+w, y2-w-1:y2+w] = label2
                    img[x3-w-1:x3+w, y3-w-1:y3+w] = label3
                    img[x4-w-1:x4+w, y4-w-1:y4+w] = label4
                    
                    priority1 = get_priority(img, i1, j1, window_size, label1)
                    priority2 = get_priority(img, i2, j2, window_size, label2)
                    priority3 = get_priority(img, i3, j3, window_size, label3)
                    priority4 = get_priority(img, i4, j4, window_size, label4)
                
                    if priority1 != -1:
                            priority_queue.put((priority1, (window_size, i1, j1)))
                    if priority2 != -1:
                            priority_queue.put((priority2, (window_size, i2, j2)))
                    if priority3 != -1:
                            priority_queue.put((priority3, (window_size, i3, j3)))
                    if priority4 != -1:
                            priority_queue.put((priority4, (window_size, i4, j4)))
            
            print(f"Iteration: {iteration}, Computational budget: {COMPUTATIONAL_BUDGET}, priority queue size: {priority_queue.qsize()}")
            
            #self.console.log(f"Iteration: {iteration}, Computational budget: {COMPUTATIONAL_BUDGET}, priority queue size: {priority_queue.qsize()}")
            iteration += 1
        print(f"Iteration: {iteration}, Computational budget: {COMPUTATIONAL_BUDGET}, priority queue size: {priority_queue.qsize()}")
            
        return img
        
        
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