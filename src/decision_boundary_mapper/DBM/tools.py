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
    if j - w > 0:
        neighbors.append(img[i, j - w])
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

@jit
def get_pixel_priority(img, i, j, window_width, window_height, label):
    resolution = img.shape[0]
    # getting the 4 neighbors
    i, j = int(i), int(j)
    w = int(window_width / 2)
    h = int(window_height / 2)
    neighbors = []
    
    if i - h > 0:
        neighbors.append(img[i - h, j])
    if i + h + 1 < resolution:
        neighbors.append(img[i + h + 1, j])
    
    if j - w > 0:
        neighbors.append(img[i , j - w])
    if j + w + 1 < resolution:
        neighbors.append(img[i, j + w + 1])
        
    cost = 0
    for neighbor in neighbors:
        if neighbor != label:
            cost += 1
    
    if cost == 0:
        return -1
    
    cost /= len(neighbors)
    cost *= window_width * window_height      
    
    return 1/cost


def get_confidence_based_split(img, conf_img, i, j, w, h):
    resolution = img.shape[0]
    initial_i, initial_j = i, j
    i, j = int(i), int(j)
    dw = int(w / 2)
    dh = int(h / 2)
    label = img[i,j]
    conf = conf_img[i,j]

    neighbors = []
    
    if i - dh > 0:
        new_i = i - dh
        if img[new_i, j] != label:
            neighbors.append((new_i, -1, conf_img[new_i, j]))
    if i + dh + 1 < resolution:
        new_i = i + dh + 1
        if img[new_i, j] != label:
            neighbors.append((new_i, -1, conf_img[new_i, j]))
    
    if j - dw > 0:
        new_j = j - dw
        if img[i, new_j] != label:
            neighbors.append((-1, new_j, conf_img[i, new_j]))
    if j + dw + 1 < resolution:
        new_j = j + dw + 1 
        if img[i, new_j] != label:
            neighbors.append((-1, new_j, conf_img[i, new_j]))
    
        
    splits_x = []
    splits_y = []

    for (y,x,c) in neighbors:
        if x == -1:
            split = get_split_position(initial_i, conf, y, -c)
            if split is not None:
                splits_y.append(split)
        if y == -1:
            split = get_split_position(initial_j, conf, x, -c)
            if split is not None:
                splits_x.append(split)
     
    if len(splits_x) == 0 and len(splits_y) == 0:
        splits_x = [j + 1]
        splits_y = [i + 1]        
    
    splits_x = [j - dw + 1] + splits_x + [j + dw + 1]
    splits_y = [i - dh + 1] + splits_y + [i + dh + 1]
    
    representatives = []
    sizes = []
    markers_x = []
    
    for index in range(len(splits_x) - 1):  
        w = splits_x[index + 1] - splits_x[index]
        x = (splits_x[index + 1] + splits_x[index] - 1) / 2
        if w == 0:
            w = 1
            x = splits_x[index]
        markers_x.append((x,w))
            
    for index in range(len(splits_y) - 1):   
        h = splits_y[index + 1] - splits_y[index]
        y = (splits_y[index + 1] + splits_y[index] - 1) / 2
        if h == 0:
            h = 1
            y = splits_y[index]
        for (x,w) in markers_x:
            representatives.append((x,y))
            sizes.append((w,h))    
    
    return representatives, sizes

def get_split_position(x1, c1, x2, c2):
    A = np.array([[x1,1],[x2,1]])
    B = np.array([c1,c2])
    coefficients = np.linalg.solve(A,B)
    boundary = round(- coefficients[1] / coefficients[0])
    if boundary <= x2 + 1 and x2 < x1:
        return None
    if boundary >= x2 - 1 and x2 > x1:
        return None
    return boundary

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
