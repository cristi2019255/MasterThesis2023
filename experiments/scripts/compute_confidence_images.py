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
from scipy import interpolate
from .config import INTERPOLATION_FOLDER_NAME, CONFIDENCE_MAP_FOLDER_NAME

FOLDER = os.path.join("experiments", "results", "MNIST", "DBM", "t-SNE", "binary_split")
INTERPOLATION_METHOD = "linear"
RESOLUTIONS = [250, 500, 1000]

def generate_interpolated_image(sparse_map, resolution:int, method:str='linear'):
        """
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


def compute_confidence_images(folder=FOLDER, interpolation_method=INTERPOLATION_METHOD, resolutions=RESOLUTIONS):
    save_folder = os.path.join(folder, INTERPOLATION_FOLDER_NAME, interpolation_method)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    confidence_maps_folder = os.path.join(folder, CONFIDENCE_MAP_FOLDER_NAME)
    for resolution in resolutions:
        print("Computing confidence image for resolution: ", resolution)
        confidence_map_path = os.path.join(confidence_maps_folder, str(resolution) + ".npy")
        with open(confidence_map_path, "rb") as f:
            confidence_map = np.load(f)
        print("Confidence map uploaded successfully")
        
        interpolated_confidence_img = generate_interpolated_image(confidence_map, resolution, interpolation_method).T

        save_path = os.path.join(save_folder, str(resolution) + ".npy")
        with open(save_path, "wb") as f:
            np.save(f, interpolated_confidence_img)
        print("Confidence image saved successfully")
    