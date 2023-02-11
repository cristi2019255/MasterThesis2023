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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

def transform_tsne(X: np.ndarray):
    """ Transforms the given data to 2D using t-SNE.

    Args:
        X (np.ndarray): The data to be transformed
    """
    X_flat = X.reshape(X.shape[0], -1)
    projection = TSNE(n_components=2, random_state=0)
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

def transform_umap(X: np.ndarray):
    """ Transforms the given data to 2D using UMAP.

    Args:
        X (np.ndarray): The data to be transformed
    """
    X_flat = X.reshape(X.shape[0], -1)
    projection = UMAP(n_components=2, random_state=0)
    X2d = projection.fit_transform(X_flat)
    return X2d

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
    error = np.sqrt(np.linalg.norm(dx)**2 + np.linalg.norm(dy)**2)
    return error

a = 50
r = 256
theta = np.radians(np.linspace(0, 360, a+1))
phi = np.radians(np.linspace(0, 360, a+1))
x = r * np.einsum("i,j->ij", np.cos(phi), np.sin(theta))
y = r * np.einsum("i,j->ij", np.sin(phi), np.sin(theta))
z = r * np.einsum("i,j->ij", np.ones(len(theta)), np.cos(theta))
x = x.flatten()
y = y.flatten() 
z = z.flatten()


Xnd = np.array([(xx, yy, zz) for xx, yy, zz in zip(x, y, z)])
print(Xnd.shape)


X2d = transform_tsne(Xnd)
with open("X2D.npy", "wb") as f:
    np.save(f, X2d)
with open("XND.npy", "wb") as f:
    np.save(f, Xnd)


with open("X2D.npy", "rb") as f:
    X2d = np.load(f)
with open("XND.npy", "rb") as f:
    Xnd = np.load(f)

import tensorflow as tf

decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),             
            tf.keras.layers.Dense(3, activation='linear'),
    ], name="decoder")

optimizer = tf.keras.optimizers.Adam()
decoder.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
decoder.fit(X2d, Xnd, epochs=50, batch_size=32, verbose=1)

resolution = 256

x = np.linspace(0, resolution-1, resolution)
y = np.linspace(0, resolution-1, resolution)
x, y = np.meshgrid(x, y)

x_max = X2d[:,0].max()
x_min = X2d[:,0].min()
y_max = X2d[:,1].max()
y_min = X2d[:,1].min()

X2d_total_space = np.array([(xx/resolution * (x_max - x_min) + x_min, yy /resolution * (y_max - y_min) + y_min) for xx, yy in zip(x.flatten(), y.flatten())])

X3d = decoder.predict(X2d_total_space)
X3d = X3d.reshape(resolution, resolution, 3)

error_img = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        error = get_inv_proj_error(i=i, j=j, Xnd=X3d)
        error_img[i,j] = error

# normalize error image
#error_img = (error_img - error_img.min()) / (error_img.max() - error_img.min())

from matplotlib.offsetbox import AnnotationBbox, TextArea

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(121)
ax2 = plt.subplot(122, projection='3d')
ax.set_aspect('equal')

def add_annotation(X2d, Xnd, direction=1, label_text="", Xnd_initial=None, color_on_sphere='red'):            
    def display_annotation(event):
        if event.inaxes == None:
            return
            
        j, i = int(event.xdata), int(event.ydata)
                
        # change the annotation box position relative to mouse.
        ws = (event.x > fig_width/2.)*-1 + (event.x <= fig_width/2.) 
        hs = (event.y > fig_height/2.)*-1 + (event.y <= fig_height/2.)
        annLabels.xybox = (-direction * 50. * ws, direction * 50. *hs)
        # make annotation box visible
        
        data = find_data_point(i, j)                            
        if data is not None:
            ax2.clear()
            ax2.scatter3D([data[0]], [data[1]], [data[2]], c=color_on_sphere, s=100, marker='*')
            ax2.scatter3D(Xnd_initial[:,0], Xnd_initial[:,1], Xnd_initial[:,2], c='gray', s=1, alpha=0.15)

            label.set_text(f"{label_text} {data}")
            annLabels.set_visible(True)
            annLabels.xy = (j, i)
            
        fig.canvas.draw_idle()        
        
    def generate_encoded_mapping(X2d, Xnd):
        mapper = {}
        for k in range(len(X2d)):
            [x, y] = X2d[k]
            x, y = int(x), int(y)
            mapper[f"{y} {x}"] = Xnd[k]
        return mapper
    
    def find_data_point(i, j):
        if f"{i} {j}" in mapper.keys():
            data = mapper[f"{i} {j}"]
            return data 

    xybox=(50., 50.)
    label = TextArea("Data point label: None")
    annLabels = AnnotationBbox(label, (0,0), xybox=xybox, xycoords='data',
                    boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

    ax.add_artist(annLabels)        
    annLabels.set_visible(False)
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    
    mapper = generate_encoded_mapping(X2d, Xnd)
                
    fig.canvas.mpl_connect('motion_notify_event', display_annotation)    
    #print(mapper)


X2d[:,0] = (X2d[:,0] - x_min) / (x_max - x_min) * (resolution -1)
X2d[:,1] = (X2d[:,1] - y_min) / (y_max - y_min) * (resolution -1)
X2d_total_space[:,0] = (X2d_total_space[:,0] - x_min) / (x_max - x_min) * (resolution -1)
X2d_total_space[:,1] = (X2d_total_space[:,1] - y_min) / (y_max - y_min) * (resolution -1)

add_annotation(X2d, Xnd, direction=1, label_text="Original data: ", Xnd_initial=Xnd, color_on_sphere='red')
add_annotation(X2d_total_space, X3d.reshape(-1,3), direction=-1, label_text="Reconstructed data: ", color_on_sphere='black', Xnd_initial=Xnd)

ax.imshow(error_img)
ax.plot(X2d[:,0], X2d[:,1] , 'bo', markersize=3)
plt.show()


