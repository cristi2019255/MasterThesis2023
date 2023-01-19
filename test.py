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

import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.getcwd(), "models", "SDBM", "fast_boundary_map.npy")
path1 = os.path.join(os.getcwd(), "models", "SDBM", "boundary_map.npy")

fig, [ax1,ax2] = plt.subplots(1, 2)
with open(path, "rb") as f:
    img1 = np.load(f)
    ax1.imshow(img1)

with open(path1, "rb") as f:
    img2 = np.load(f)
    ax2.imshow(img2)


errors = 0
for (i, j), z in np.ndenumerate(img1):
    if img2[i,j] == -1 or img2[i,j] == -2:
        continue
    if img2[i,j] != z:
       errors += 1
       print(f"Error at {i}, {j} : {img2[i,j]} != {z}")

print(f"Percentage of errors: {errors / (img1.shape[0] * img1.shape[1]) * 100} %")     
print(f"Number of errors: {errors}")

plt.show()