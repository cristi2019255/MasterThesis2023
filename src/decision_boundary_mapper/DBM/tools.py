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

@jit
def get_inv_proj_error(i:int,j:int, Xnd:np.ndarray, w:int=1, h:int=1):
    """Calculates the inverse projection error for a given point in the image.
        Args:
            i (int): the row of the point
            j (int): the column of the point
            Xnd (np.ndarray): the nD space
    """
    xl = Xnd[i,j-w] if j - w >= 0 else Xnd[i,j]
    xr = Xnd[i,j+w] if j + w < Xnd.shape[1] else Xnd[i,j]
    yl = Xnd[i-h,j] if i - h >= 0 else Xnd[i,j]
    yr = Xnd[i+h,j] if i + h < Xnd.shape[0] else Xnd[i,j]
    
    dw = 2 * w
    dh = 2 * h
    if (j - w < 0) or (j + w >= Xnd.shape[1]):
        dw = w
    if (i - h < 0) or (i + h >= Xnd.shape[0]):
        dh = h
    
    dx = (xl - xr) / dw
    dy = (yl - yr) / dh    
    return np.sqrt(np.linalg.norm(dx)**2 + np.linalg.norm(dy)**2)

@jit
def get_proj_error(indices_source: np.ndarray, indices_embedding: np.ndarray, k: int=10):
    """ Calculates the projection error for a given data point.
        Args:
            indices_source (np.ndarray): the indices of the point neighbors in the source (i.e. nD) space
            indices_embedding (np.ndarray): the indices of the point neighbors in the embedding (i.e. 2D) space
            k (int): the number of neighbors to consider
    """
    assert len(indices_source) == len(indices_embedding)
    n = len(indices_source)
    
    continuity = 0.0
    trustworthiness = 0.0
    
    # computing the continuity error
    for i in range(k):
        rank = 0
        while indices_source[i] != indices_embedding[rank]:
            rank += 1

        if rank > k:
            continuity += rank - k

    continuity = 2 * continuity / (k * (2*n - 3*k - 1))
    
    # computing the trustworthiness error
    for i in range(k):    
        rank = 0
        while indices_source[rank] != indices_embedding[i]:
            rank += 1

        if rank > k:
            trustworthiness += rank - k

    trustworthiness = 2 * trustworthiness / (k * (2*n - 3*k - 1))
    
    return (continuity + trustworthiness) / 2

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
