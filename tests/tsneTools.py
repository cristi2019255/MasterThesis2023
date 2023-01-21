# Copyright 2022 Cristian Grosu
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def get_tsne_embedding(corpus, model_name = "MNIST"): 
    # We want to get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components)
    
    tsne_result = tsne.fit_transform(corpus)
    print(tsne_result)
    # (corpus.length, 2)
    
    # save tsne embedding
    if not os.path.exists(os.path.join("data", model_name)):
        os.makedirs(os.path.join("data", model_name))
    with open(os.path.join("data", model_name, "tsne_embedding.npy"), "wb") as f:
        np.save(f, tsne_result)
    
    return tsne_result

def load_tsne_embedding(model_name):
    with open(os.path.join("data", model_name, "tsne_embedding.npy"), "rb") as f:
        tsne_embedding = np.load(f)
    return tsne_embedding

def plot_tsne_embedding(labels, model_name = "MNIST"):
    tsne_result = load_tsne_embedding(model_name)
    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette="deep", style='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()