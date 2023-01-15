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
    img = np.load(f)
    ax1.imshow(img)

with open(path1, "rb") as f:
    img = np.load(f)
    ax2.imshow(img)
    
plt.show()