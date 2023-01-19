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

path_img_fast = os.path.join(os.getcwd(), "models", "SDBM", "fast_boundary_map.npy")
path_img_real = os.path.join(os.getcwd(), "models", "SDBM", "boundary_map.npy")

path_confidence_fast = os.path.join(os.getcwd(), "models", "SDBM", "fast_boundary_map_confidence.npy")
path_confidence_real = os.path.join(os.getcwd(), "models", "SDBM", "boundary_map_confidence.npy")

with open(path_img_fast, "rb") as f:
    img1 = np.load(f)

with open(path_img_real, "rb") as f:
    img2 = np.load(f)

errors = 0
for (i, j), z in np.ndenumerate(img1):
    if img2[i,j] != z:
       errors += 1
       print(f"Error at {i}, {j} : {img2[i,j]} != {z}")

print(f"Percentage of errors: {errors / (img1.shape[0] * img1.shape[1]) * 100} %")     
print(f"Number of errors: {errors}")

with open(path_confidence_fast, "rb") as f:
    img1_conf = np.load(f)

with open(path_confidence_real, "rb") as f:
    img2_conf = np.load(f)

COLORS_MAPPER = {
    0: [1,0,0], 
    1: [0,1,0], 
    2: [0,0,1], 
    3: [1,1,0], 
    4: [0,1,1], 
    5: [1,0,1], 
    6: [0.5,0.5,0.5], 
    7: [0.5,0,0], 
    8: [0,0.5,0], 
    9: [0,0,0.5]
}

immg1 = np.zeros((img1.shape[0], img1.shape[1], 4))
immg2 = np.zeros((img2.shape[0], img2.shape[1], 4))

for (i, j), z in np.ndenumerate(img1):
    immg1[i,j] = COLORS_MAPPER[z] + [img1_conf[i,j]]

for (i, j), z in np.ndenumerate(img2):
    immg2[i,j] = COLORS_MAPPER[z] + [img2_conf[i,j]]

fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4)
ax1.imshow(img1)
ax2.imshow(img2)
ax3.imshow(immg1)
ax4.imshow(immg2)

plt.show()