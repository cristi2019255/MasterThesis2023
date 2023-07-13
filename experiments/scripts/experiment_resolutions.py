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
from .utils import import_2d_data, import_data, import_dbm, experiment, save_result
from src import FAST_DBM_STRATEGIES, NNArchitecture
import time

# WARNING: !!! If you change the DBM_TECHNIQUE make sure the RESULTS_FOLDER set correctly !!!
# ---------------------------------------------------
# Prepare the configurations
DATASET_NAME = 'MNIST'
DBM_TECHNIQUE = 'SDBM'
PROJECTION = 'UMAP'
# ---------------------------------------------------


TMP_FOLDER = "tmp"
TRAIN_2D_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, DBM_TECHNIQUE, PROJECTION, "train_2d.npy")
TEST_2D_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, DBM_TECHNIQUE, PROJECTION, "test_2d.npy")
CLASSIFIER_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, "classifier")
LOAD_FOLDER = os.path.join(TMP_FOLDER, DATASET_NAME, DBM_TECHNIQUE)

# ---------------------------------------------------
FAST_DECODING_STRATEGY = FAST_DBM_STRATEGIES.NONE
# ---------------------------------------------------

RESOLUTION_RANGE = (50, 2000, 50)

#RESULTS_FOLDER = os.path.join("experiments", "results", DATASET_NAME, DBM_TECHNIQUE, PROJECTION, FAST_DECODING_STRATEGY.value)
RESULTS_FOLDER = os.path.join("experiments", "results", DATASET_NAME, DBM_TECHNIQUE, FAST_DECODING_STRATEGY.value) # the results folder for SDBM
CONFIDENCE_SUBFOLDER = os.path.join(RESULTS_FOLDER, "confidence")
CONFIDENCE_MAP_SUBFOLDER = os.path.join(RESULTS_FOLDER, "confidence_map")
IMG_SUBFOLDER = os.path.join(RESULTS_FOLDER, "img")
EXPERIMENT_METADATA_PATH = os.path.join(RESULTS_FOLDER, "experiment_metadata.txt")
EXPERIMENT_RESULTS_PATH = os.path.join(RESULTS_FOLDER, "experiment_results.txt")

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# ---------------------------------------------------
def create_results_folder():
    # create the results folder
    if not os.path.exists(IMG_SUBFOLDER):
        os.makedirs(IMG_SUBFOLDER)
    if not os.path.exists(CONFIDENCE_SUBFOLDER):
        os.makedirs(CONFIDENCE_SUBFOLDER)
    if FAST_DECODING_STRATEGY != FAST_DBM_STRATEGIES.NONE and not os.path.exists(CONFIDENCE_MAP_SUBFOLDER):
        os.makedirs(CONFIDENCE_MAP_SUBFOLDER)

# ---------------------------------------------------
@experiment(EXPERIMENT_METADATA_PATH)
def resolutions_run_times():
    create_results_folder()
    
    if len(os.listdir(IMG_SUBFOLDER)) != 0 or len(os.listdir(CONFIDENCE_SUBFOLDER)) != 0:
        print("WARNING: The experiment was already run. If you want to run it again, please delete the folder: ", RESULTS_FOLDER)
        print("WARNING: Skipping the experiment...")
        return
    # ---------------------------------------------------
    
    # Prepare the data    
    X_train, X_test, Y_train, Y_test = import_data(dataset_name=DATASET_NAME)
    
    # Prepare the DBM
    dbm = import_dbm(dbm_technique=DBM_TECHNIQUE, classifier_path=CLASSIFIER_PATH)
    
    # Run the generation of the boundary map first time to upload the decoding model    
    if DBM_TECHNIQUE == 'DBM':
        X2d_train, X2d_test = import_2d_data(train_2d_path=TRAIN_2D_PATH, test_2d_path=TEST_2D_PATH)
        dbm.generate_boundary_map(X_train,
                                X_test,
                                X2d_train,
                                X2d_test,
                                resolution=10,
                                fast_decoding_strategy=FAST_DBM_STRATEGIES.NONE,
                                load_folder=LOAD_FOLDER,
                                projection=PROJECTION
                            )
    else:
        dbm.generate_boundary_map(X_train, Y_train,
                                  X_test, Y_test,
                                  nn_architecture=NNArchitecture.SSNP,
                                  load_folder=LOAD_FOLDER,
                                  resolution=10
                                )

    DECODER = {
        FAST_DBM_STRATEGIES.BINARY: dbm._get_img_dbm_fast_,
        FAST_DBM_STRATEGIES.CONFIDENCE_BASED: dbm._get_img_dbm_fast_confidences_strategy,
        FAST_DBM_STRATEGIES.CONFIDENCE_INTERPOLATION: dbm._get_img_dbm_fast_confidence_interpolation_strategy,
        FAST_DBM_STRATEGIES.NONE: dbm._get_img_dbm_,
    }
    
    
    # create the experiment metadata file
    with open(EXPERIMENT_METADATA_PATH, "a") as f:
        f.write("RESOLUTION_RANGE: " + str(RESOLUTION_RANGE) + "\n")
        f.write("FAST_DECODING_STRATEGY: " + FAST_DECODING_STRATEGY.value + "\n")
        f.write("\n\n\n")
    with open(EXPERIMENT_RESULTS_PATH, "w") as f:
        f.write("RESOLUTION,TIME\n")
    
    # ---------------------------------------------------
    # ---------------------------------------------------
    # Run the generation of the boundary map for different resolutions
    for resolution in range(*RESOLUTION_RANGE):
        start = time.time()
        if FAST_DECODING_STRATEGY != FAST_DBM_STRATEGIES.CONFIDENCE_INTERPOLATION:
            img, img_confidence, confidence_map = DECODER[FAST_DECODING_STRATEGY](resolution)
        else: 
            img, img_confidence, confidence_map = dbm._get_img_dbm_fast_confidence_interpolation_strategy(resolution, initial_resolution = 32)
        end = time.time()
        decoding_time = round(end - start, 3)
        print("Resolution: ", resolution, "Decoding time: ", decoding_time)
        # ---------------------------------------------------
        # Save the results
        with open(EXPERIMENT_RESULTS_PATH, "a") as f:
            f.write(str(resolution) + "," + str(decoding_time) + "\n")
            
        img_path = os.path.join(IMG_SUBFOLDER, str(resolution) + ".npy")
        img_confidence_path = os.path.join(CONFIDENCE_SUBFOLDER, str(resolution) + ".npy")
        
        save_result(img_path, img)
        save_result(img_confidence_path, img_confidence)
        
        if confidence_map is not None:
            confidence_map_path = os.path.join(CONFIDENCE_MAP_SUBFOLDER, str(resolution) + ".npy")
            save_result(confidence_map_path, confidence_map)