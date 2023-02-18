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

with open("continuity_errors_sparse.npy", "rb") as f:
    continuity_errors_sparse = np.load(f)

resolution = 256
X, Y, Z = [], [], []
for (x, y, z) in continuity_errors_sparse:
    X.append(x)
    Y.append(y)
    Z.append(z)
#X, Y, Z = np.array(X), np.array(Y), np.array(Z)


ti = np.linspace(0, resolution - 1, resolution)
xx, yy = np.meshgrid(ti, ti)

x = X
y = Y
z = Z
rbf = Rbf(x, y, z) # fails due to Matrix is singular. ???

zz = rbf(xx, yy)

jet = cm = plt.get_cmap('jet')

plt.pcolor(xx, yy, zz, cmap=jet)
plt.colorbar()



plot3 = plt.plot(X, Y, 'ko', markersize=2)  # the original points.

plt.show()