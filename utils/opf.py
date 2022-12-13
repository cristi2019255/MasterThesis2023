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

import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SemiSupervisedOPF
import numpy as np

def opf(X_train, Y_train):
    # Splitting data into training and validation sets
    #X_train, X_unlabeled, Y_train, Y_unlabeled = s.split(
    #    X_train, Y_train, percentage=0.25, random_state=1
    #)
    pr_x = int(0.1*len(X_train))
    pr_y = int(0.1*len(Y_train))
    X_train, X_unlabeled, Y_train, Y_unlabeled = X_train[:pr_x], X_train[pr_x:], Y_train[:pr_y], Y_train[pr_y:]

    # Creates a SemiSupervisedOPF instance
    opf = SemiSupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

    # Fits training data along with unlabeled data into the semi-supervised classifier
    opf.fit(X_train, Y_train, X_unlabeled)
    
    # Predicts new data
    preds = opf.predict(X_unlabeled)
    
    # Calculating accuracy
    acc = g.opf_accuracy(Y_unlabeled, preds)
    print(f"Accuracy: {acc}")

    return np.hstack((Y_train, preds))