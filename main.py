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

import os
from tests import test_projection_errors
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # uncomment to disable GPU, run on CPU only 
from utils.dataReader import import_mnist_dataset

SAMPLES_LIMIT = 5000
    
def main():
    # import MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    
    # limiting to first 1000 samples for testing
    X_train = X_train[:int(0.7 * SAMPLES_LIMIT)]
    Y_train = Y_train[:int(0.7 * SAMPLES_LIMIT)]
    X_test = X_test[:int(0.3 * SAMPLES_LIMIT)]
    Y_test = Y_test[:int(0.3 * SAMPLES_LIMIT)]
    
    pass

def test():
    test_projection_errors()

if __name__ == '__main__':
    #main()
    test()   