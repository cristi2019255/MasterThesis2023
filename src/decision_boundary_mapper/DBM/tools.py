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

from math import ceil, floor
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

        indices[i] = np.argsort(dist_vector)[1:]  # exclude the point itself

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
    return np.sqrt(np.linalg.norm(dx)**2 + np.linalg.norm(dy)**2)

@njit(parallel=True)
def get_proj_error_parallel(indices_source: np.ndarray, indices_embedding: np.ndarray, k: int = 10):
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
def get_proj_error(indices_source: np.ndarray, indices_embedding: np.ndarray, k: int = 10):
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
def get_pixel_priority(img, i, j, window_width, window_height, label):
    """
       Calculates the priority of decoding a chunk of pixels.
       The chunk is defined by the window size and the pixel coordinates (i,j).

    Args:
        img (np.ndarray): the image w.r.t. which the priority is calculated
        i (float): the row of the current pixel
        j (float): the column of the current pixel
        window_width (float), window_height (float): the window size (i.e. the chunk size)
        label (any): the label of the current pixel

    Returns:
        priority (float): the priority of decoding the chunk in range [0,1] or -1 if the chunk does not need to be decoded
    """
    resolution = img.shape[0]
    # getting the 4 neighbors
    w = (window_width - 1) / 2
    h = (window_height - 1)/ 2
    neighbors = []

    if i - h - 1 >= 0:
        neighbors.append(img[int(i - h - 1), int(j)])
    if i + h + 1 < resolution:
        neighbors.append(img[int(i + h + 1), int(j)])

    if j - w - 1 >= 0:
        neighbors.append(img[int(i), int(j - w - 1)])
    if j + w + 1 < resolution:
        neighbors.append(img[int(i), int(j + w + 1)])

    cost = 0
    for neighbor in neighbors:
        if neighbor != label:
            cost += 1


    cost /= len(neighbors)
    cost *= window_width * window_height
    
    if cost == 0:
        return -1

    return 1/cost

@njit
def binary_split(i, j, W, H):
    Wc, Wf = ceil(W/2), floor(W/2)
    Hc, Hf = ceil(H/2), floor(H/2)
    slack_w = 0 if Wc == Wf else 0.5
    slack_h = 0 if Hc == Hf else 0.5
    c_i = i + slack_h
    c_j = j + slack_w
    representatives = [(c_j - Wc / 2, c_i - Hc / 2), 
                       (c_j - Wc / 2, c_i + Hf / 2), 
                       (c_j + Wf / 2, c_i - Hc / 2),
                       (c_j + Wf / 2, c_i + Hf / 2)]
    sizes = [(Wc, Hc), (Wc, Hf), (Wf, Hc), (Wf, Hf)]
    return representatives, sizes

@njit(parallel=True)
def get_projection_errors_using_inverse_projection(Xnd: np.ndarray, X2d: np.ndarray, spaceNd: np.ndarray, space2d: np.ndarray, progress, k: int = 10):

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

@njit
def generate_windows(window_size: int, initial_resolution: int, resolution: int = 1024):

    indexes = [((i * window_size + window_size / 2 - 0.5), (j * window_size + window_size / 2 - 0.5)) for i in range(initial_resolution) for j in range(initial_resolution)]
    sizes = [(window_size, window_size)] * len(indexes)
    if initial_resolution * window_size == resolution:
        return indexes, sizes, initial_resolution
    
    slack = (resolution - initial_resolution * window_size) / 2
        
    indexes += [(i * window_size + window_size / 2 - 0.5, resolution - slack - 0.5) for i in range(initial_resolution)]
    sizes += [(window_size, (resolution - initial_resolution * window_size))] * initial_resolution
    indexes += [(resolution - slack - 0.5, i * window_size + window_size / 2 - 0.5) for i in range(initial_resolution)]
    sizes += [((resolution - initial_resolution * window_size), window_size)] * initial_resolution
    indexes += [(resolution - slack - 0.5, resolution - slack - 0.5)]
    sizes += [((resolution - initial_resolution * window_size), (resolution - initial_resolution * window_size))]
    initial_resolution += 1
        
    return indexes, sizes, initial_resolution

@njit
def get_window_borders(x, y, w, h):
    # returns the borders of the window by its center and size
    # x, y - center of the window
    # w, h - size of the window
    # returns left, right, top, bottom
    return int(x - (w - 1) / 2), int(x + (w - 1) / 2), int(y - (h - 1) / 2), int(y + (h - 1) / 2)

def get_tasks_with_same_priority(priority_queue):
    # take the highest priority task
    priority, item = priority_queue.get()
    # getting all the items with the same priority
    items = [item]
    if priority_queue.empty():
        return items
        
    next_priority, next_item = priority_queue.get()
    while priority == next_priority:
        items.append(next_item)
        if priority_queue.empty():
            break
        next_priority, next_item = priority_queue.get()
        
    # putting back the last item in the queue
    if priority != next_priority:
        priority_queue.put((next_priority, next_item))
            
    return items

@njit 
def get_split_position(x1: float, x2:float, bound: float, c11: float, c12: float, c21: float, c22: float) -> int | None:
    assert(x1 != x2)

    a, b = (c11 - c12) / (x1 - x2), c11 - x1 * ((c11 - c12) / (x1 - x2)) 
    c, d = (c21 - c22) / (x1 - x2), c21 - x1 * ((c21 - c22) / (x1 - x2))

    if a == c:
        return None
    
    boundary = round((d - b) / (a - c))

    if (bound < x1) and (bound + 1 < boundary < x1):
        return boundary
    if (bound > x1) and (x1 < boundary < bound - 1):
        return boundary
    
    return None

def get_confidence_splits(img, conf_img, img_indexes, c_y, c_x, top, bottom, left, right):
    resolution = img.shape[0]
    i, j = int(c_y), int(c_x)
    label = img[i, j]
    c11 = conf_img[i, j][label]

    splits_x, splits_y = [], []    
    
    vertical = [top] if top >= 0 else []
    vertical += [bottom] if bottom < resolution else []
    horizontal = [left] if left >= 0 else []
    horizontal += [right] if right < resolution else []
    
    for k in vertical:
        new_label = img[k, j]
        (x, y) = img_indexes[k, j]
        if new_label != label:
            split = get_split_position(c_y, y, k, c11, conf_img[y, j][label], conf_img[i, j][new_label], conf_img[y, j][new_label])
            if split is not None:
                splits_y.append(split)

    for k in horizontal:
        new_label = img[i, k]
        (x, y) = img_indexes[i, k]
        if new_label != label:
            split = get_split_position(c_x, x, k, c11, conf_img[i, x][label], conf_img[i, j][new_label], conf_img[i, x][new_label])
            if split is not None:
                splits_x.append(split)
                
    return splits_x, splits_y

def get_confidence_based_split(img, conf_img, img_indexes, i, j, W, H):
    c_y, c_x = i, j
    left, right, top, bottom = get_window_borders(c_x, c_y, W, H)
    # going to the nearest border outside of the window
    left -= 1
    right += 1
    top -= 1
    bottom += 1
    splits_x, splits_y = get_confidence_splits(img, conf_img, img_indexes, c_y, c_x, top, bottom, left, right)
    
    if len(splits_x) == 0 and len(splits_y) == 0:
        return binary_split(c_y, c_x, W, H)
    
    return confidence_based_split(splits_x, splits_y, c_y, c_x, W, H, top, bottom, left, right)

def confidence_based_split(splits_x, splits_y, c_y, c_x, W, H, top, bottom, left, right):
    markers_x, markers_y = [], []
    
    if len(splits_x) == 0:
        Wc, Wf = ceil(W/2), floor(W/2)
        slack_w = 0 if Wc == Wf else 0.5
        c_j = c_x + slack_w
        markers_x = [(c_j - Wc / 2, Wc), (c_j + Wf / 2, Wf)]    
        
    if len(splits_y) == 0:
        Hc, Hf = ceil(H/2), floor(H/2)
        slack_h = 0 if Hc == Hf else 0.5
        c_i = c_y + slack_h
        markers_y = [(c_i - Hc / 2, Hc), (c_i + Hf / 2, Hf)]
    
    if len(splits_x) == 1:
        x = splits_x[0]
        markers_x = [((x + left + 1) / 2, abs(x - left)), ((x + right) / 2, abs(x - right) - 1)]
    if len(splits_y) == 1:
        y = splits_y[0]
        markers_y = [((y + top + 1) / 2, abs(y - top)), ((y + bottom) / 2, abs(y - bottom) - 1)]
        
    if len(splits_x) == 2:
        x1, x2 = splits_x
        markers_x = [((x1 + left + 1) / 2, abs(x1 - left)), ((x1 + x2) / 2, abs(x1 - x2) - 1), ((x2 + right - 1) / 2, abs(x2 - right))]
    if len(splits_y) == 2:
        y1, y2 = splits_y
        markers_y = [((y1 + top + 1) / 2, abs(y1 - top)), ((y1 + y2) / 2, abs(y1 - y2) - 1), ((y2 + bottom - 1) / 2, abs(y2 - bottom))]
        
    representatives, sizes = [], []
    for (y, h) in markers_y:
        for (x, w) in markers_x:
            representatives.append((x, y))
            sizes.append((w, h))
    
    return representatives, sizes
