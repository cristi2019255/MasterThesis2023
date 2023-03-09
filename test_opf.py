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

from src.decision_boundary_mapper.utils import import_mnist_dataset
from tests import opf
import numpy as np

(X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
Y_train = Y_train[:3500]
X_train = X_train[:3500]

#with open("/home/cristi/codeRepo/MasterThesis2023/tmp/MNIST/DBM/t-SNE/train_2d.npy", "rb") as f:
#    X_train = np.load(f)

print(X_train.shape)

opf(X_train, Y_train)