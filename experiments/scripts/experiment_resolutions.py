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
from .utils import import_2d_data, import_data, import_dbm, is_experiment, save_result
from src import FAST_DBM_STRATEGIES
import time

# WARNING: !!! for now we only use the DBM !!!
# ---------------------------------------------------
# Prepare the configurations
DATASET_NAME = 'MNIST'
DBM_TECHNIQUE = 'DBM'
PROJECTION = 't-SNE'
    
TRAIN_2D_PATH = os.path.join("tmp", DATASET_NAME, DBM_TECHNIQUE, PROJECTION, "train_2d.npy")
TEST_2D_PATH = os.path.join("tmp", DATASET_NAME, DBM_TECHNIQUE, PROJECTION, "test_2d.npy")
CLASSIFIER_PATH = os.path.join("tmp", DATASET_NAME, "classifier")
LOAD_FOLDER = os.path.join("tmp", DATASET_NAME, DBM_TECHNIQUE)
FAST_DECODING_STRATEGY = FAST_DBM_STRATEGIES.NONE
RESOLUTION_RANGE = (50, 2000, 50)

RESULTS_FOLDER = os.path.join("experiments", "results", DATASET_NAME, DBM_TECHNIQUE, PROJECTION, str(FAST_DECODING_STRATEGY))
CONFIDENCE_SUBFOLDER = os.path.join(RESULTS_FOLDER, "confidence")
IMG_SUBFOLDER = os.path.join(RESULTS_FOLDER, "img")
EXPERIMENT_METADATA_PATH = os.path.join(RESULTS_FOLDER, "experiment_metadata.txt")

# ---------------------------------------------------
@is_experiment
def resolutions_run_times():
   
    # ---------------------------------------------------
    
    # Prepare the data    
    X_train, X_test, _, _ = import_data(dataset_name=DATASET_NAME)
    X2d_train, X2d_test = import_2d_data(train_2d_path=TRAIN_2D_PATH, test_2d_path=TEST_2D_PATH)
    
    # Prepare the DBM
    dbm = import_dbm(dbm_technique=DBM_TECHNIQUE, classifier_path=CLASSIFIER_PATH)
    
  
    # Run the generation of the boundary map first time to upload the decoding model
    dbm.generate_boundary_map(X_train,
                              X_test,
                              X2d_train,
                              X2d_test,
                              resolution=10,
                              fast_decoding_strategy=FAST_DBM_STRATEGIES.NONE,
                              load_folder=LOAD_FOLDER,
                              projection='t-SNE')

    
    DECODER = {
        FAST_DBM_STRATEGIES.BINARY: dbm._get_img_dbm_fast_,
        FAST_DBM_STRATEGIES.CONFIDENCE_BASED: dbm._get_img_dbm_fast_confidences_strategy,
        FAST_DBM_STRATEGIES.HYBRID: dbm._get_img_dbm_fast_hybrid_strategy,
        FAST_DBM_STRATEGIES.NONE: dbm._get_img_dbm_,
    }
    
    
    # create the results folder
    if not os.path.exists(IMG_SUBFOLDER):
        os.makedirs(IMG_SUBFOLDER)
    if not os.path.exists(CONFIDENCE_SUBFOLDER):
        os.makedirs(CONFIDENCE_SUBFOLDER)
    # create the experiment metadata file
    with open(EXPERIMENT_METADATA_PATH, "w") as f:
        f.write("RESOLUTION_RANGE: " + str(RESOLUTION_RANGE) + "\n")
        f.write("FAST_DECODING_STRATEGY: " + str(FAST_DECODING_STRATEGY) + "\n")
        f.write("\n\n\n")
        f.write("RESOLUTION \t & \t TIME \n")
    
    # ---------------------------------------------------
    # ---------------------------------------------------
    # Run the generation of the boundary map for different resolutions
    for resolution in range(*RESOLUTION_RANGE):
        start = time.time()
        img, img_confidence = DECODER[FAST_DECODING_STRATEGY](resolution)
        end = time.time()
        decoding_time = round(end - start, 3)
        print("Resolution: ", resolution, "Decoding time: ", decoding_time)
        # ---------------------------------------------------
        # Save the results
        with open(EXPERIMENT_METADATA_PATH, "a") as f:
            f.write(str(resolution) + "\t & \t" + str(decoding_time) + "\n")
            
        img_path = os.path.join(IMG_SUBFOLDER, str(resolution) + ".npy")
        img_confidence_path = os.path.join(CONFIDENCE_SUBFOLDER, str(resolution) + "_confidence.npy")
        save_result(img_path, img)
        save_result(img_confidence_path, img_confidence)
