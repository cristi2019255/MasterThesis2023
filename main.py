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

import tensorflow as tf
import os
import numpy as np
import random
from src import DBM_usage_example, SDBM_usage_example, DBM_usage_example_GUI, SDBM_usage_example_GUI
from tests.DBM.test import test_dbm, test_projection_errors


SEED = 42


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


# Call the above function with seed value
set_global_determinism(seed=SEED)


def main():
    # DBM_usage_example()
    DBM_usage_example_GUI()
    # SDBM_usage_example()
    # SDBM_usage_example_GUI()
    # test_projection_errors()
    # test_dbm()


if __name__ == '__main__':
    main()
