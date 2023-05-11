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
from math import ceil, floor
from queue import PriorityQueue
from scipy import interpolate
import dask.array as da
from sklearn.neighbors import KDTree
from numba_progress import ProgressBar
import tensorflow as tf
from tqdm import tqdm
from enum import Enum

from .tools import get_confidence_based_split, get_decode_pixel_priority, get_inv_proj_error, get_nd_indices_parallel, euclidean, get_pixel_priority, get_proj_error_parallel, get_projection_errors_using_inverse_projection, get_tasks_with_same_priority
from .AbstractNN import AbstractNN
from ..utils import track_time_wrapper, INVERSE_PROJECTION_ERRORS_FILE, PROJECTION_ERRORS_INTERPOLATED_FILE, PROJECTION_ERRORS_INVERSE_PROJECTION_FILE
from ..Logger import Logger, LoggerInterface

DBM_DEFAULT_CHUNK_SIZE = 10000
DBM_DEFAULT_RESOLUTION = 256
DBM_IMAGE_NAME = "boundary_map"
DBM_CONFIDENCE_IMAGE_NAME = "boundary_map_confidence"

PROJECTION_ERRORS_NEIGHBORS_NUMBER = 10

time_tracker_console = Logger(name="Decision Boundary Mapper - DBM", info_color="cyan", show_init=False)

class FAST_DBM_STRATEGIES(Enum):
    NONE = "none"
    BINARY = "binary_split"
    CONFIDENCE_BASED = "confidence_split"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class AbstractDBM:
    """ 
    Decision Boundary Mapper Abstract class

    Public methods that can be used:
        refit_classifier \n
        save_classifier  \n
        load_classifier  \n
        get_dbm          \n  
        generate_inverse_projection_errors \n
        generate_projection_errors         \n

    Methods to be implemented by the class that implements this class:
        _predict2dspace_ (X)

    Example of usage: 
        class SuperBoundaryMapper(AbstractDBM):
            def _predict2dspace_ (X):
                # To be implemented
    """

    def __init__(self, classifier, logger: LoggerInterface | None = None):
        """
        Initializes the classifier and the logger

        Args:
            classifier (tf.keras.Model): The classifier to be used
            logger (LoggerInterface, optional): The logger for the outputting info messages. Defaults to None.
        """
        if logger is None:
            self.console = Logger(name="Decision Boundary Mapper - DBM")
        else:
            self.console = logger

        self.console.log("Loaded classifier: " + classifier.name + "")
        self.classifier = classifier
        self.neural_network: AbstractNN
        self.X2d: np.ndarray
        self.Xnd: np.ndarray

    def refit_classifier(self, Xnd: np.ndarray, Y: np.ndarray, save_folder: str, epochs: int = 2, batch_size: int = 32):
        """ 
        Refits the classifier on the given data set.

        Args:             
            Xnd (np.ndarray): The data set
            Y (np.ndarray): The new labels
            save_folder (str): Saving folder of the classifier
            epochs (int, optional): Number of epochs. Defaults to 2.
            batch_size (int): Number of training samples to be taken in a single batch. Defaults to 32.
        """
        self.console.log(f"Refiting classifier for {epochs} epochs and batch size {batch_size}, please wait...")
        self.classifier.fit(Xnd, Y, epochs=epochs, batch_size=batch_size, verbose=0)  # type: ignore
        self.console.log("Finished refitting classifier")
        self.save_classifier(save_folder=save_folder)

    def save_classifier(self, save_folder: str):
        """ 
        Saves a copy of the classifier.

        Args:
            save_folder (str): The folder where the classifier will be saved
        """
        self.console.log("Saving the classifier...")
        self.classifier.save(save_folder, save_format="tf")  # type: ignore
        self.console.log("A copy of the classifier was saved!")

    def load_classifier(self, load_folder: str):
        """ 
        Loads a copy of the classifier.

        Args:
            load_folder (str): The folder where the classifier is saved
        """
        self.classifier = tf.keras.models.load_model(load_folder)

    def _predict2dspace_(self, X2d: np.ndarray | list[tuple[float, float]]) -> tuple:
        """ 
        Predicts the labels for the given 2D data set.
        IMPORTANT: This method should be implemented by the classes that implement this class

        Args:
            X2d (np.ndarray): The 2D data set

        Returns:
            predicted_labels (np.array): The predicted labels for the given 2D data set
            predicted_confidences (np.array): The predicted probabilities for the given 2D data set, for each data point the confidence is returned (i.e. the maximum probability)
            predictions (np.array): The predicted probabilities for the given 2D data set, for each data point a list of probabilities is returned
        """
        return None, None, None

    def get_dbm(self, fast_decoding_strategy: FAST_DBM_STRATEGIES, resolution: int, load_folder: str) -> tuple:
        """
        Delegates the generation of the DBM to the according functionality based on the fast_decoding_strategy

        Args:
            fast_decoding_strategy (FAST_DBM_STRATEGIES): The strategy to use for the generation of the DBM
            resolution (int): The desired resolution of the DBM image
            load_folder (str): The folder in which we save the results

        Returns:
            img (np.ndarray): The DBM image
            img_confidence (np.ndarray): The DBM confidence image
        """
        save_img_path = os.path.join(load_folder, DBM_IMAGE_NAME)
        save_img_confidence_path = os.path.join(load_folder, DBM_CONFIDENCE_IMAGE_NAME)

        match fast_decoding_strategy:
            case FAST_DBM_STRATEGIES.NONE:
                img, img_confidence = self._get_img_dbm_(resolution)
            case FAST_DBM_STRATEGIES.BINARY:
                save_img_path += f"_fast_{FAST_DBM_STRATEGIES.BINARY.value}"
                save_img_confidence_path += f"_fast_{FAST_DBM_STRATEGIES.BINARY.value}"
                img, img_confidence = self._get_img_dbm_fast_(resolution)
            case FAST_DBM_STRATEGIES.CONFIDENCE_BASED:
                save_img_path += f"_fast_{FAST_DBM_STRATEGIES.CONFIDENCE_BASED.value}"
                save_img_confidence_path += f"_fast_{FAST_DBM_STRATEGIES.CONFIDENCE_BASED.value}"
                img, img_confidence = self._get_img_dbm_fast_confidences_strategy(resolution)

        with open(f"{save_img_path}.npy", 'wb') as f:
            np.save(f, img)  # type: ignore
        with open(f"{save_img_confidence_path}.npy", 'wb') as f:
            np.save(f, img_confidence)  # type: ignore

        return img, img_confidence  # type: ignore

    @track_time_wrapper(logger=time_tracker_console)
    def _get_img_dbm_(self, resolution: int):
        """ 
        This function generates the 2D image of the boundary map using the trained neural network and the classifier.

        Args:
            resolution (int): The resolution of the 2D image to be generated 

        Returns:
            img (np.array): The 2D image of the boundary map
            img_confidence (np.array): The confidence of map of each pixel of the 2D image of the boundary map
            
        Example:
            >>> img, img_confidence = self._get_img_dbm_(resolution = 100)
        """
        space2d = np.array([(i / resolution, j / resolution) for i in range(resolution) for j in range(resolution)])
        self.console.log("Predicting labels for the 2D boundary mapping using the nD data and the trained classifier...")

        chunk_size = DBM_DEFAULT_CHUNK_SIZE
        chunks = (resolution * resolution) // chunk_size + 1
        space2d_chunks = np.array_split(space2d, chunks)

        img, img_confidence = np.array([]), np.array([])

        chunk_index = 0
        for space2d_chunk in space2d_chunks:
            chunk_index += 1
            self.console.log(f"Predicting labels for the 2D boundary mapping using the nD data and the trained classifier... (chunk {chunk_index}/{len(space2d_chunks)})")
            predicted_labels, predicted_confidence, _ = self._predict2dspace_(space2d_chunk)
            img = np.concatenate((img, predicted_labels))
            img_confidence = np.concatenate((img_confidence, predicted_confidence))

        img = img.reshape((resolution, resolution))
        img_confidence = img_confidence.reshape((resolution, resolution))

        return (img, img_confidence)

    @track_time_wrapper(logger=time_tracker_console)
    def _get_img_dbm_fast_(self, resolution: int, computational_budget=None, interpolation_method: str = "linear"):
        """
        This function generates the 2D image of the boundary map. It uses a fast algorithm to generate the image.

        ATTENTION: Highly recommended to use resolution = 2 ^ n, where n is an integer number (powers of 2).

        Args:
            resolution (int): the resolution of the 2D image to be generated
            computational_budget (int, optional): The computational budget to be used. Defaults to None.
            interpolation_method (str, optional): The interpolation method to be used for the interpolation of sparse data generated by the fast algorithm.
                                                  Defaults to "linear". The options are: "nearest", "linear", "cubic"
        Returns:
            img, img_confidence: The 2D image of the boundary map and a image with the confidence for each pixel
            spaceNd: The nD space points
        Example:
            >>> img, img_confidence = self._get_img_dbm_fast_(resolution=32, computational_budget=1000)
        """
        # ------------------------------------------------------------
        # Setting the initial parameters
        WINDOW_SIZE = 8
        NEIGHBORS_NUMBER = 4

        assert (resolution > WINDOW_SIZE)

        INITIAL_RESOLUTION = resolution // WINDOW_SIZE

        if computational_budget is None:
            computational_budget = resolution * resolution

        assert (computational_budget > INITIAL_RESOLUTION * INITIAL_RESOLUTION)
        INITIAL_COMPUTATIONAL_BUDGET = computational_budget

        if (resolution % INITIAL_RESOLUTION != 0):
            self.console.warn(f"The required resolution is not a multiple of the initial window size ({WINDOW_SIZE} x {WINDOW_SIZE})")
            self.console.log("The resolution will be set to the closest multiple of the window size")
            resolution = INITIAL_RESOLUTION * WINDOW_SIZE
            self.resolution = resolution
            self.console.log(f"Resolution was set to {resolution} x {resolution}")

        window_size = WINDOW_SIZE
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        img = np.zeros((resolution, resolution))
        priority_queue = PriorityQueue()
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # generate the initial points
        self.console.log(f"Generating the initial central points within each window... total number of windows ({(INITIAL_RESOLUTION * INITIAL_RESOLUTION)})")

        space2d = [((i * window_size + window_size / 2 - 0.5) / resolution, (j * window_size + window_size / 2 - 0.5) / resolution) for i in range(INITIAL_RESOLUTION) for j in range(INITIAL_RESOLUTION)]

        predicted_labels, predicted_confidence, _ = self._predict2dspace_(space2d)

        predictions = predicted_labels.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        confidences = predicted_confidence.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))

        # initialize the nD space

        computational_budget -= INITIAL_RESOLUTION * INITIAL_RESOLUTION

        # fill the initial points in the 2D image
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x0, x1, y0, y1 = i * window_size, (i+1) * window_size, j * window_size, (j+1) * window_size
                img[x0:x1, y0:y1] = predictions[i, j]
        # -------------------------------------

        # collecting the indexes of the actual computed pixels in the 2D image and the confidence of each pixel
        # creating an artificial border for the 2D confidence image
        confidence_map = []

        if interpolation_method != "nearest":
            border_indices = [(i * window_size + window_size / 2 - 0.5, 0) for i in range(INITIAL_RESOLUTION)] + \
                [(i * window_size + window_size / 2 - 0.5, resolution - 1) for i in range(INITIAL_RESOLUTION)] + \
                [(0, i * window_size + window_size / 2 - 0.5) for i in range(-1, INITIAL_RESOLUTION+1)] + \
                [(resolution - 1, i * window_size + window_size / 2 - 0.5) for i in range(-1, INITIAL_RESOLUTION + 1)]

            space2d_border = [(i / resolution, j / resolution) for (i, j) in border_indices]
            _, confidences_border, _ = self._predict2dspace_(space2d_border)
            computational_budget -= len(space2d_border)
            confidence_map = [(i, j, conf) for (i, j), conf in zip(
                border_indices, confidences_border)]

        # analyze the initial points and generate the priority queue
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x, y = i * window_size + window_size / 2 - 0.5, j * window_size + window_size / 2 - 0.5
                confidence_map.append((x, y, confidences[i, j]))
                priority = get_decode_pixel_priority(img, x, y, window_size, predictions[i, j])
                if priority != -1:
                    priority_queue.put((priority, (window_size, x, y)))

        # -------------------------------------
        # start the iterative process of filling the image
        self.console.log(f"Starting the iterative process of refining windows...")
        while computational_budget > 0 and not priority_queue.empty():
            # take the highest priority tasks
            items = get_tasks_with_same_priority(priority_queue)

            space2d, indices = [], []
            single_points_space, single_points_indices = [], []
            window_sizes = []

            for (window_size, i, j) in items:
                if window_size == 1:
                    single_points_indices += [(int(i), int(j))]
                    single_points_space += [(i / resolution, j / resolution)]
                    continue

                window_sizes += [window_size / 2] * NEIGHBORS_NUMBER
                w = window_size / NEIGHBORS_NUMBER
                neighbors = [(i - w, j - w), (i - w, j + w), (i + w, j - w), (i + w, j + w)]
                space2d += [(x / resolution, y / resolution) for (x, y) in neighbors]
                indices += neighbors

            space = space2d + single_points_space

            # check if the computational budget is enough and update it
            if computational_budget - len(space) < 0:
                self.console.warn("Computational budget exceeded, stopping the process")
                break

            computational_budget -= len(space)

            # decode the space
            predicted_labels, predicted_confidence, _ = self._predict2dspace_(space)
            # copy the image to a new one, the new image will be updated with the new labels, the old one will be used to calculate the priorities
            new_img = np.copy(img)

            # fill the new image with the single points
            single_points_labels = predicted_labels[len(space2d):]
            single_points_confidences = predicted_confidence[len(space2d):]

            predicted_labels = predicted_labels[:len(space2d)]
            predicted_confidence = predicted_confidence[:len(space2d)]

            # fill the new image with the new labels and update the priority queue
            for w, (x, y), label, conf in zip(window_sizes, indices, predicted_labels, predicted_confidence):
                # all the pixels in the window are set to the same label
                # starting from the top left corner of the window
                # ending at the bottom right corner of the window
                # the +1 is because the range function is not inclusive
                confidence_map.append((x, y, conf))
                x0, x1, y0, y1 = ceil(x-w/2), floor(x + w/2), ceil(y - w/2), floor(y + w/2)
                new_img[x0:x1 + 1, y0:y1 + 1] = label

            for i in range(len(single_points_indices)):
                new_img[single_points_indices[i]] = single_points_labels[i]
                confidence_map.append((single_points_indices[i][0], single_points_indices[i][1], single_points_confidences[i]))

            # update the priority queue after the window has been filled
            for w, (x, y), label in zip(window_sizes, indices, predicted_labels):
                priority = get_decode_pixel_priority(new_img, x, y, w, label)
                if priority != -1:
                    priority_queue.put((priority, (w, x, y)))

            # update the image
            img = new_img

        # summary
        self.console.log(f"Finished decoding the image, initial computational budget: {INITIAL_COMPUTATIONAL_BUDGET} computational budget left: {computational_budget}")
        self.console.log(f"Items left in the priority queue: {priority_queue.qsize()}")

        # generating the confidence image using interpolation based on the confidence map
        img_confidence = self._generate_interpolated_image_(sparse_map=confidence_map,
                                                            resolution=resolution,
                                                            method=interpolation_method).T

        return img, img_confidence

    @track_time_wrapper(logger=time_tracker_console)
    def _get_img_dbm_fast_confidences_strategy(self, resolution: int, computational_budget=None, interpolation_method: str = "linear"):
        """
        This function generates the 2D image of the boundary map. It uses a fast algorithm that uses a confidence based strategy to generate the image.

        Args:
            resolution (int): the resolution of the 2D image to be generated
            computational_budget (int, optional): The computational budget to be used. Defaults to None.
            interpolation_method (str, optional): The interpolation method to be used for the interpolation of sparse data generated by the fast algorithm.
                                                  Defaults to "linear". The options are: "nearest", "linear", "cubic"
        Returns:
            img, img_confidence: The 2D image of the boundary map and a image with the confidence for each pixel
            spaceNd: The nD space points
        Example:
            >>> img, img_confidence = self._get_img_dbm_fast_confidences_strategy(resolution=32, computational_budget=1000)
        """
        # ------------------------------------------------------------
        # Setting the initial parameters
        INITIAL_RESOLUTION = 32

        if computational_budget is None:
            computational_budget = resolution * resolution

        assert (resolution > INITIAL_RESOLUTION)
        assert (computational_budget > INITIAL_RESOLUTION * INITIAL_RESOLUTION)

        INITIAL_COMPUTATIONAL_BUDGET = computational_budget

        # ------------------------------------------------------------
        if (resolution % INITIAL_RESOLUTION != 0):
            resolution = INITIAL_RESOLUTION * (resolution // INITIAL_RESOLUTION)
            self.resolution = resolution
            self.console.log(f"Resolution was set to {resolution} x {resolution}")

        window_size_w = resolution // INITIAL_RESOLUTION
        window_size_h = resolution // INITIAL_RESOLUTION

        # ------------------------------------------------------------
        img = np.zeros((resolution, resolution), dtype=np.uint8)
        priority_queue = PriorityQueue()
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # generate the initial points
        self.console.log(f"Generating the initial central points within each window... total number of windows ({(INITIAL_RESOLUTION * INITIAL_RESOLUTION)})")

        space2d = [((i * window_size_w + window_size_w / 2 - 0.5) / resolution, (j * window_size_h + window_size_h / 2 - 0.5) / resolution) for i in range(INITIAL_RESOLUTION) for j in range(INITIAL_RESOLUTION)]

        predicted_labels, predicted_confidence, predicted_confidences = self._predict2dspace_(space2d)
        
        num_classes = len(predicted_confidences[0])
        pseudo_conf_img = np.zeros((resolution, resolution, num_classes))
      

        predictions = predicted_labels.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        confidences = predicted_confidence.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION))
        nd_confidences = predicted_confidences.reshape((INITIAL_RESOLUTION, INITIAL_RESOLUTION, num_classes))
        # initialize the nD space

        computational_budget -= INITIAL_RESOLUTION * INITIAL_RESOLUTION

        # fill the initial points in the 2D image
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                x0, x1, y0, y1 = i * window_size_w, (i+1) * window_size_w, j * window_size_h, (j+1) * window_size_h
                img[x0:x1, y0:y1] = predictions[i, j]
                pseudo_conf_img[x0:x1, y0:y1, :] = nd_confidences[i, j]
        # -------------------------------------

        # collecting the indexes of the actual computed pixels in the 2D image and the confidence of each pixel
        # creating an artificial border for the 2D confidence image
        confidence_map = []

        if interpolation_method != "nearest":
            border_indices = [(i * window_size_h + window_size_h / 2 - 0.5, 0) for i in range(INITIAL_RESOLUTION)] + \
                [(i * window_size_h + window_size_h / 2 - 0.5, resolution - 1) for i in range(INITIAL_RESOLUTION)] + \
                [(0, i * window_size_h + window_size_h / 2 - 0.5) for i in range(-1, INITIAL_RESOLUTION+1)] + \
                [(resolution - 1, i * window_size_h + window_size_h / 2 - 0.5) for i in range(-1, INITIAL_RESOLUTION + 1)]

            space2d_border = [(i / resolution, j / resolution) for (i, j) in border_indices]
            _, confidences_border, _ = self._predict2dspace_(space2d_border)
            computational_budget -= len(space2d_border)
            confidence_map = [(i, j, conf) for (i, j), conf in zip(border_indices, confidences_border)]

        # analyze the initial points and generate the priority queue
        for i in range(INITIAL_RESOLUTION):
            for j in range(INITIAL_RESOLUTION):
                y, x = i * window_size_w + window_size_w / 2 - 0.5, j * window_size_h + window_size_h / 2 - 0.5
                confidence_map.append((y, x, confidences[i, j]))
                priority = get_pixel_priority(img, y, x, window_size_w, window_size_h, predictions[i, j])
                if priority != -1:
                    priority_queue.put((priority, (window_size_w, window_size_h, y, x)))

        # -------------------------------------
        # start the iterative process of filling the image
        self.console.log(f"Starting the iterative process of refining windows...")

        iteration = 0
        while computational_budget > 0 and not priority_queue.empty():
            iteration += 1
            # take the highest priority tasks
            items = get_tasks_with_same_priority(priority_queue)

            space2d, indices = [], []
            single_points_space, single_points_indices = [], []
            window_sizes = []

            for (w, h, i, j) in items:
                if w == 1 and h == 1:
                    single_points_indices += [(int(i), int(j))]
                    single_points_space += [(i / resolution, j / resolution)]
                    continue

                neighbors, sizes = get_confidence_based_split(img, pseudo_conf_img, i, j, w, h)
                space2d += [(y / resolution, x / resolution) for (x, y) in neighbors]
                window_sizes += sizes
                indices += neighbors

            space = space2d + single_points_space

            # check if the computational budget is enough and update it
            if computational_budget - len(space) < 0:
                self.console.warn("Computational budget exceeded, stopping the process")
                break

            computational_budget -= len(space)

            # decode the space
            predicted_labels, predicted_confidence, predicted_confidences = self._predict2dspace_(space)
            # copy the image to a new one, the new image will be updated with the new labels, the old one will be used to calculate the priorities
            new_img = np.copy(img)
            new_pseudo_conf_image = pseudo_conf_img

            # fill the new image with the single points
            single_points_labels = predicted_labels[len(space2d):]
            single_points_confidences = predicted_confidence[len(space2d):]
            single_points_confidences_nd = predicted_confidences[len(space2d):]

            for i in range(len(single_points_indices)):
                new_img[single_points_indices[i]] = single_points_labels[i]
                new_pseudo_conf_image[single_points_indices[i]] = single_points_confidences_nd[i]
                confidence_map.append((single_points_indices[i][0], single_points_indices[i][1], single_points_confidences[i]))

            predicted_labels = predicted_labels[:len(space2d)]
            predicted_confidence = predicted_confidence[:len(space2d)]
            predicted_confidences = predicted_confidences[:len(space2d)]

            # fill the new image with the new labels and update the priority queue
            for (w, h), (x, y), label, conf, confs in zip(window_sizes, indices, predicted_labels, predicted_confidence, predicted_confidences):
                # all the pixels in the window are set to the same label
                # starting from the top left corner of the window
                # ending at the bottom right corner of the window
                # the +1 is because the range function is not inclusive
                confidence_map.append((y, x, conf))
                x0, x1, y0, y1 = ceil(x - w / 2), floor(x + w / 2), ceil(y - h / 2), floor(y + h / 2)
                new_img[y0:y1 + 1, x0:x1 + 1] = label
                new_pseudo_conf_image[y0:y1 + 1, x0:x1 + 1, :] = confs

            # update the priority queue after the window has been filled
            for (w, h), (x, y), label in zip(window_sizes, indices, predicted_labels):
                priority = get_pixel_priority(new_img, y, x, w, h, label)
                if priority != -1:
                    priority_queue.put((priority, (w, h, y, x)))

            # update the image
            img = new_img
            pseudo_conf_img = new_pseudo_conf_image

        # summary
        self.console.log(f"Finished decoding the image, initial computational budget: {INITIAL_COMPUTATIONAL_BUDGET} computational budget left: {computational_budget}")
        self.console.log(f"Items left in the priority queue: {priority_queue.qsize()}")

        # generating the confidence image using interpolation based on the confidence map
        img_confidence = self._generate_interpolated_image_(sparse_map=confidence_map,
                                                            resolution=resolution,
                                                            method=interpolation_method).T

        return img, img_confidence

    @track_time_wrapper(logger=time_tracker_console)
    def generate_inverse_projection_errors(self, resolution: int, save_folder: str | None = None):
        """ 
        Calculates the inverse projection errors of the given data.

        Args:
            resolution (int): The resolution of the errors image to generate
            save_folder (str): The path of the folder in which we want to save the results. Defaults to None

        Returns:
            errors (np.ndarray): The inverse projection errors matrix of the given data. (resolution x resolution)
        """

        self.console.log("Calculating the inverse projection errors of the given data")
        errors = np.zeros((resolution, resolution))

        w, h = 1, 1

        current_row = np.array([(0, j / resolution) for j in range(resolution)])
        next_row = np.array([( 1 / resolution, j / resolution) for j in range(resolution)])
        data_2d = np.concatenate((current_row, next_row))
        data_nd = self.neural_network.decode(data_2d)

        previous_row_nd = None
        current_row_nd: np.ndarray = data_nd[:resolution]
        next_row_nd: np.ndarray = data_nd[resolution:]

        for i in tqdm(range(resolution)):
            for j in range(resolution):
                dw = w if (j - w < 0) or (j + w >= resolution) else 2 * w
                dh = h if (i - h < 0) or (i + h >= resolution) else 2 * h
                xnd = current_row_nd[j]
                xl = current_row_nd[j-w] if j - w >= 0 else xnd
                xr = current_row_nd[j+w] if j + w < resolution else xnd
                yl = previous_row_nd[j] if previous_row_nd is not None else xnd
                yr = next_row_nd[j] if next_row_nd is not None else xnd

                dx = (xl - xr) / dw
                dy = (yl - yr) / dh
                errors[i, j] = get_inv_proj_error(dx, dy)

            previous_row_nd = current_row_nd
            current_row_nd = next_row_nd
            if i + 2 < resolution:
                next_row = np.array([((i + 2) / resolution, j / resolution) for j in range(resolution)])
                next_row_nd = self.neural_network.decode(next_row)
            else:
                next_row_nd = None  # type: ignore

        # normalizing the errors to be in the range [0,1]
        errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))

        if save_folder is not None:
            self.console.log("Saving the inverse projection errors results")
            save_path = os.path.join(save_folder, INVERSE_PROJECTION_ERRORS_FILE)
            with open(save_path, "wb") as f:
                np.save(f, errors)
            self.console.log("Saved inverse projection errors results!")

        return errors

    def _generate_interpolated_image_(self, sparse_map, resolution:int, method:str='linear'):
        """
        A private method that uses interpolation to generate the values for the 2D space image
           The sparse map is a list of tuples (x, y, data)
           The sparse map represents a structured but non uniform grid of data values
           Therefore usual rectangular interpolation methods are not suitable
           For the interpolation we use the scipy.interpolate.griddata function with the linear method
        Args:
            sparse_map (list): a list of tuples (x, y, data) where x and y are the coordinates of the pixel and data is the data value
            resolution (int): the resolution of the image we want to generate (the image will be a square image)
            method (str, optional): The method to be used for the interpolation. Defaults to 'linear'. Available methods are: 'nearest', 'linear', 'cubic'

        Returns:
            np.array: an array of shape (resolution, resolution) containing the data values for the 2D space image
        """
        X, Y, Z = [], [], []
        for (x, y, z) in sparse_map:
            X.append(x)
            Y.append(y)
            Z.append(z)
        X, Y, Z = np.array(X), np.array(Y), np.array(Z)
        xi = np.linspace(0, resolution-1, resolution)
        yi = np.linspace(0, resolution-1, resolution)
        return interpolate.griddata((X, Y), Z, (xi[None, :], yi[:, None]), method=method)

    @track_time_wrapper(logger=time_tracker_console)
    def _generate_interpolation_rbf_(self, sparse_map, resolution, function='linear'):
        """A private method that uses interpolation to generate the values for the 2D space image
           The sparse map is a list of tuples (x, y, data) where x, y and data are in the range [0, 1]

        Args:
            sparse_map (np.ndarray): a list of tuples (x, y, data) where x and y are the coordinates of the pixel and data is the data value
            resolution (int): the resolution of the image we want to generate (the image will be a square image)
            function (str, optional): Defaults to 'euclidean'.
        """
        self.console.log(
            "Computing the interpolated image using RBF interpolation...")
        X, Y, Z = [], [], []
        for (x, y, z) in sparse_map:
            X.append(x)
            Y.append(y)
            Z.append(z)

        rbf = interpolate.Rbf(X, Y, Z, function=function)
        ti = np.linspace(0, 1, resolution)
        xx, yy = np.meshgrid(ti, ti)
        
        """
            using dask to parallelize the computation of the rbf function
            the rbf function is applied to each pixel of the image
            the image is divided into chunks and each chunk is processed in parallel
            the result is then merged together
            the number of chunks is equal to the number of cores
        """
        cores = 4

        ix = da.from_array(xx, chunks=(1, cores))  # type: ignore
        iy = da.from_array(yy, chunks=(1, cores))  # type: ignore
        iz = da.map_blocks(rbf, ix, iy)            # type: ignore
        zz = iz.compute()
        self.console.log("Finished computing the interpolated image using RBF interpolation")
        return zz

    def generate_projection_errors(self, 
                                   Xnd: np.ndarray | None = None,
                                   X2d: np.ndarray | None = None,
                                   resolution: int | None = None,
                                   use_interpolation: bool = True,
                                   save_folder: str | None = None):
        """ 
        Calculates the projection errors of the given data.

        Args:
            Xnd (np.array): The data to be projected. The data must be in the range [0,1].
            X2d (np.array): The 2D projection of the data. The data must be in the range [0,1].
            resolution (int): The resolution of the 2D space.
            spaceNd (np.array): The nD space of the data.
            use_interpolation (bool): Whether to use interpolation to generate the projection errors or use the inverse projection. Defaults to interpolation usage
            save_folder (str): The folder path where to store the projection errors results. Defaults to None.
        Returns:
            errors (np.array): The projection errors matrix of the given data. 

        Example:
            >>> from decision-boundary-mapper import SDBM
            >>> classifier = ...
            >>> Xnd = ...
            >>> X2d = ...
            >>> sdbm = SDBM(classifier)
            >>> errors = sdbm.get_projection_errors(Xnd, X2d)
            >>> plt.imshow(errors)
            >>> plt.show()
            >>> ...
            >>> spaceNd = ...
            >>> errors = sdbm.get_projection_errors(Xnd, X2d, spaceNd=spaceNd, use_interpolation=False)
            >>> plt.imshow(errors)
            >>> plt.show()
        """

        if Xnd is None:
            if self.Xnd is None:
                self.console.error("No nD data provided and no data stored in the DBM object.")
                raise ValueError("No nD data provided and no data stored in the DBM object.")
            Xnd = self.Xnd
        if X2d is None:
            if self.X2d is None:
                self.console.error("No 2D data provided and no data stored in the DBM object.")
                raise ValueError("No 2D data provided and no data stored in the DBM object.")
            X2d = self.X2d
        if resolution is None:
            if self.resolution is None:
                self.console.error("The resolution of the 2D space is not set, try to call the method 'generate_boundary_map' first.")
                raise Exception("The resolution of the 2D space is not set, try to call the method 'generate_boundary_map' first.")
            resolution = self.resolution

        assert len(X2d) == len(Xnd)
        X2d = X2d.reshape((X2d.shape[0], -1))
        Xnd = Xnd.reshape((Xnd.shape[0], -1))

        assert X2d.shape[1] == 2

        self.console.log("Calculating the projection errors of the given data, this might take a couple of minutes. Please wait...")
        if use_interpolation:
            errors = self._generate_projection_errors_using_interpolation_(Xnd, X2d, resolution)
        else:
            errors = self._generate_projection_errors_using_inverse_projection_(Xnd, X2d, resolution)

        self.console.log("Finished computing the projection errors!")
        if save_folder is not None:
            self.console.log("Saving the projection errors results")
            save_path = os.path.join(save_folder, PROJECTION_ERRORS_INTERPOLATED_FILE) if use_interpolation else os.path.join(save_folder, PROJECTION_ERRORS_INVERSE_PROJECTION_FILE)
            with open(save_path, "wb") as f:
                np.save(f, errors)
            self.console.log("Saved projection errors results!")

        return errors

    @track_time_wrapper(logger=time_tracker_console)
    def _generate_projection_errors_using_interpolation_(self, Xnd, X2d, resolution):
        errors = np.zeros((resolution, resolution))
        K = PROJECTION_ERRORS_NEIGHBORS_NUMBER  # Number of nearest neighbors to consider when computing the errors
        metric = "euclidean"

        self.console.log("Computing the 2D tree")
        tree = KDTree(X2d, metric=metric)
        self.console.log("Finished computing the 2D tree")
        self.console.log("Computing the 2D tree indices")
        indices_embedded = tree.query(X2d, k=len(X2d), return_distance=False)
        # Drop the actual point itself
        indices_embedded = indices_embedded[:, 1:]
        self.console.log("Finished computing the 2D tree indices")

        self.console.log("Calculating the nD distance indices")
        indices_source = get_nd_indices_parallel(Xnd, metric=euclidean)
        self.console.log("Finished computing the nD distance indices")

        sparse_map = []
        for k in range(len(X2d)):
            x, y = X2d[k]
            sparse_map.append((x, y, get_proj_error_parallel(indices_source[k], indices_embedded[k], k=K)))

        errors = self._generate_interpolation_rbf_(sparse_map, resolution, function='linear').T

        # resize the errors in range [0,1]
        errors = (errors - errors.min()) / (errors.max() - errors.min())

        return errors

    @track_time_wrapper(logger=time_tracker_console)
    def _generate_projection_errors_using_inverse_projection_(self,
                                                              Xnd: np.ndarray,
                                                              X2d: np.ndarray,
                                                              resolution: int = 256):

        K = PROJECTION_ERRORS_NEIGHBORS_NUMBER  # Number of nearest neighbors to consider when computing the errors
        space2d = np.array([(i / resolution, j / resolution) for i in range(resolution) for j in range(resolution)])  # generate the 2D flatten space

        errors = np.array([])
        # split space2d into chunks
        # split the space into chunks of 10000 points max
        chunks_number = resolution * resolution // DBM_DEFAULT_CHUNK_SIZE + 1
        self.console.log(f"Splitting the 2D space into {chunks_number} chunks")
        space2d_chunks = np.array_split(space2d, chunks_number)

        chunk_index = 0
        for space2d_chunk in space2d_chunks:
            chunk_index += 1
            self.console.log(f"Computing the projection errors for chunk ({chunk_index}/{chunks_number}) of size: {(len(space2d_chunk))}")
            spaceNd_chunk = self.neural_network.decode(space2d_chunk)  # decode the 2D space to nD space
            spaceNd_chunk = spaceNd_chunk.reshape((spaceNd_chunk.shape[0], -1))  # flatten the space

            with ProgressBar(total=len(space2d_chunk)) as progress:
                errors_chunk = get_projection_errors_using_inverse_projection(Xnd=Xnd, X2d=X2d, spaceNd=spaceNd_chunk, space2d=space2d_chunk, k=K, progress=progress)

            errors = np.concatenate((errors, errors_chunk))

        # resize the errors in range [0,1]
        errors = (errors - errors.min()) / (errors.max() - errors.min())
        errors = errors.reshape((resolution, resolution))

        return errors
