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
from numba import jit

def get_inv_proj_error(i:int,j:int, Xnd:np.ndarray):
    """Calculates the inverse projection error for a given point in the image.
        Args:
            i (int): the row of the point
            j (int): the column of the point
            Xnd (np.ndarray): the nD space
    """
    error = 0
    # getting the neighbours of the given point
    neighbors_nd = []
    current_point_nd = Xnd[i,j]
    
    if i - 1 >= 0:
        neighbors_nd.append(Xnd[i-1,j])
    if i + 1 < Xnd.shape[0]:
        neighbors_nd.append(Xnd[i+1,j])
    if j - 1 >= 0:
        neighbors_nd.append(Xnd[i,j-1])
    if j + 1 < Xnd.shape[1]:
        neighbors_nd.append(Xnd[i,j+1])
    
    if i - 1 >= 0 and j - 1 >= 0:
        neighbors_nd.append(Xnd[i-1,j-1])
    if i - 1 >= 0 and j + 1 < Xnd.shape[1]:
        neighbors_nd.append(Xnd[i-1,j+1])
    if i + 1 < Xnd.shape[0] and j - 1 >= 0:
        neighbors_nd.append(Xnd[i+1,j-1])
    if i + 1 < Xnd.shape[0] and j + 1 < Xnd.shape[1]:
        neighbors_nd.append(Xnd[i+1,j+1])
    
    # calculating the error
    for neighbor_nd in neighbors_nd:
        error += np.linalg.norm(current_point_nd - neighbor_nd)
    
    error /= len(neighbors_nd)
    return error

def get_proj_error(indices_source: np.ndarray, indices_embedding: np.ndarray):
    """Calculates the projection error for a given data point.
        Args:
            indices_source (np.ndarray): the indices of the point neighbors in the source (i.e. nD) space
            indices_embedding (np.ndarray): the indices of the point neighbors in the embedding (i.e. 2D) space
    """
    rank = np.sum(indices_source != indices_embedding)         
    return rank / len(indices_source)

@jit
def get_decode_pixel_priority(img, i, j, window_size, label):
    """Calculates the priority of decoding a chunk of pixels.
       The chunk is defined by the window size and the pixel coordinates (i,j).

    Args:
        img (np.ndarray): the image w.r.t. which the priority is calculated
        i (float): the row of the current pixel
        j (float): the column of the current pixel
        window_size (float): the window size (i.e. the chunk size)
        label (any): the label of the current pixel

    Returns:
        priority (float): the priority of decoding the chunk in range [0,1]
    """
    resolution = img.shape[0]
    # getting the 4 neighbors
    i, j = int(i), int(j)
    w = int(window_size / 2)
    neighbors = []
    if i - w > 0:
        neighbors.append(img[i - w, j])
    if i + w + 1 < resolution:
        neighbors.append(img[i + w + 1, j])
    if j - w > 0 and i + 1 < resolution:
        neighbors.append(img[i + 1, j - w])
    if j + w + 1 < resolution:
        neighbors.append(img[i, j + w + 1])
        
    cost = 0
    for neighbor in neighbors:
        if neighbor != label:
            cost += 1
        
    if cost == 0:
        return -1
    
    cost /= len(neighbors)
    cost *= window_size      
    
    return 1/cost
