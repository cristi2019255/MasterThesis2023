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
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import dask.array as da

jet = cm = plt.get_cmap('jet')
plt.axes().set_aspect('equal')

with open("continuity_sparse_map.npy", "rb") as f:
    continuity_errors_sparse = np.load(f)
with open("trustworthiness_sparse_map.npy", "rb") as f:
    trustworthiness_errors_sparse = np.load(f)

def euclidean_norm_numpy(a, b):
    return np.linalg.norm(a - b, axis=0)

resolution = 300
X, Y, Z = [], [], []
mapper = {}
print(continuity_errors_sparse.shape)


for (x, y, z) in continuity_errors_sparse:
    if (x, y) in mapper:
        mapper[(x, y)] = (mapper[(x, y)][0] + z, mapper[(x, y)][1] + 1)
    else:
        mapper[(x, y)] = (z, 1)

for (x, y), (z, count) in mapper.items():
    X.append(x)
    Y.append(y)
    Z.append(z / count)


x, y, z = X, Y, Z
rbf = Rbf(x, y, z) 
print("RBF computed.")

ti = np.linspace(0, resolution - 1, resolution)
xx, yy = np.meshgrid(ti, ti)
n = xx.shape[1]
ix = da.from_array(xx, chunks=(1, n))
iy = da.from_array(yy, chunks=(1, n))
iz = da.map_blocks(rbf, ix, iy)
zz = iz.compute()
print("Interpolation computed.")

plt.pcolor(xx, yy, zz, cmap=jet)
plt.colorbar()

with open("zz.npy", "wb") as f:
    np.save(f, zz)

#plot3 = plt.plot(x , y, 'ko', markersize=2)  # the original points.
plt.show()


#ax = plt.axes(projection='3d')
#ax.scatter3D(x, y, z, c=z)
#plt.show()