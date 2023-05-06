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


def generate_transformation(projection: TSNE | PCA | UMAP):
    """
    Generates a 2D projection functionality
    
    Returns:
        function: X: (nd data) -> 2d projection
    """
    def transform(X: np.ndarray):
        """ 
        Transforms the given data to 2D using the projection method.

        Args:
            X (np.ndarray): The data to be transformed
        """
        X_flat = X.reshape(X.shape[0], -1)
        X2d = projection.fit_transform(X_flat)
        return X2d
    
    return transform

PROJECTION_METHODS = {
    "t-SNE": generate_transformation(TSNE(n_components=2, random_state=0, learning_rate="auto", init="random")),
    "PCA": generate_transformation(PCA(n_components=2, random_state=0)),
    "UMAP": generate_transformation(UMAP(n_components=2, random_state=0)),
    "CUSTOM": None,
}
