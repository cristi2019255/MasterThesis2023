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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np

def transform_tsne(X: np.ndarray):
    """ Transforms the given data to 2D using t-SNE.

    Args:
        X (np.ndarray): The data to be transformed
    """
    X_flat = X.reshape(X.shape[0], -1)
    projection = TSNE(n_components=2, random_state=0, learning_rate="auto", init="random")
    X2d = projection.fit_transform(X_flat)
    return X2d

def transform_umap(X: np.ndarray):
    """ Transforms the given data to 2D using UMAP.

    Args:
        X (np.ndarray): The data to be transformed
    """
    X_flat = X.reshape(X.shape[0], -1)
    projection = UMAP(n_components=2, random_state=0)
    X2d = projection.fit_transform(X_flat)
    return X2d


def transform_pca(X: np.ndarray):
    """ Transforms the given data to 2D using PCA.

    Args:
        X (np.ndarray): The data to be transformed
    """
    X_flat = X.reshape(X.shape[0], -1)
    projection = PCA(n_components=2, random_state=0)
    X2d = projection.fit_transform(X_flat)
    return X2d

PROJECTION_METHODS = {
    "t-SNE": transform_tsne,
    "PCA": transform_pca,
    "UMAP": transform_umap
}
