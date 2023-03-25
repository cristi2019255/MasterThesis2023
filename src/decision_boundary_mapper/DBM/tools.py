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
from numba import jit, njit, prange

@njit(parallel=True)
def get_nd_indices_parallel(X_nd, metric):
    """ Generates the indices of the nearest neighbors for each point in the nD space.
        Args:
            X_nd (np.ndarray): the nD dataset
            metric (function): the metric to use for calculating the distances
    """
    n_samples = X_nd.shape[0]
    dist_vector = np.zeros(n_samples, dtype=np.float64)
    indices = np.zeros((n_samples, n_samples - 1), dtype=np.int64)
    
    for i in range(n_samples):
        for j in prange(n_samples):
            dist_vector[j] = metric(X_nd[i], X_nd[j])

        indices[i] = np.argsort(dist_vector)[1:] # exclude the point itself
    
    return indices

@njit()
def generate_point_indices(X, point, metric):
    """ Generates the indices of the nearest neighbors for a point.
        Args:
            X (np.ndarray): the (Nd or 2d) dataset
            point (np.ndarray): the (Nd or 2d) point
            metric (function): the metric to use for calculating the distances
    """
    data_samples = X.shape[0]
    dist_vector = np.zeros(data_samples, dtype=np.float64)
        
    for j in range(data_samples):
        dist_vector[j] = metric(point, X[j])

    return np.argsort(dist_vector)
    
@njit(fastmath=True)
def euclidean(x, y):
    r"""Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

@jit
def get_inv_proj_error(dx, dy):
    """ TODO: add docstring
    """  
    return np.sqrt(np.linalg.norm(dx)**2 + np.linalg.norm(dy)**2)

@njit(parallel=True)
def get_proj_error_parallel(indices_source: np.ndarray, indices_embedding: np.ndarray, k: int=10):
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
    
    # computing the continuity and trustworthiness errors for each point in parallel
    for i in prange(k):
        rank_2d = 0
        while indices_source[i] != indices_embedding[rank_2d]:
            rank_2d += 1

        rank_nd = 0
        while indices_source[rank_nd] != indices_embedding[i]:
            rank_nd += 1

        if rank_2d > k:
            continuity += rank_2d - k

        if rank_nd > k:
            trustworthiness += rank_nd - k


    continuity = 2 * continuity / (k * (2*n - 3*k - 1))
    trustworthiness = 2 * trustworthiness / (k * (2*n - 3*k - 1))
    
    return (continuity + trustworthiness) / 2

@njit()
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
    
    # computing the continuity and trustworthiness errors for each point in parallel
    for i in range(k):
        rank_2d = 0
        while indices_source[i] != indices_embedding[rank_2d]:
            rank_2d += 1

        rank_nd = 0
        while indices_source[rank_nd] != indices_embedding[i]:
            rank_nd += 1

        if rank_2d > k:
            continuity += rank_2d - k

        if rank_nd > k:
            trustworthiness += rank_nd - k


    continuity = 2 * continuity / (k * (2*n - 3*k - 1))
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

@njit(parallel=True)
def get_projection_errors_using_inverse_projection( Xnd: np.ndarray, X2d: np.ndarray, spaceNd: np.ndarray, space2d: np.ndarray, progress, k: int=10):  
    
    n_points = len(space2d)                 
    data_samples = len(X2d)
    errors = np.zeros(n_points, dtype=np.float64) 
    
    for index in prange(n_points):              
        indices_embedded, indices_source = np.zeros(data_samples, dtype=np.int64), np.zeros(data_samples, dtype=np.int64)          
        indices_embedded = generate_point_indices(X2d, space2d[index], metric=euclidean)
        indices_source = generate_point_indices(Xnd, spaceNd[index], metric=euclidean)             
        errors[index] = get_proj_error(indices_source, indices_embedded, k=k)          
        progress.update(1)
    
    return errors        
