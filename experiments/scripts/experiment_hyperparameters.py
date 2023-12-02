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
from .utils import import_2d_data, import_data, import_dbm, experiment, save_result, compute_error
from src import FAST_DBM_STRATEGIES, NNArchitecture
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# WARNING: !!! If you change the DBM_TECHNIQUE make sure the RESULTS_FOLDER set correctly !!!
# ---------------------------------------------------
# Prepare the configurations
DATASET_NAME = 'MNIST'
DBM_TECHNIQUE = 'SDBM'
PROJECTION = 't-SNE'
SDBM_TECHNIQUE = 'autoencoder'
# ---------------------------------------------------


TMP_FOLDER = "tmp"
TRAIN_2D_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, DBM_TECHNIQUE, PROJECTION, "train_2d.npy")
TEST_2D_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, DBM_TECHNIQUE, PROJECTION, "test_2d.npy")
CLASSIFIER_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, "classifier")
LOAD_FOLDER = os.path.join(TMP_FOLDER, DATASET_NAME, DBM_TECHNIQUE)

# ---------------------------------------------------
FAST_DECODING_STRATEGY = FAST_DBM_STRATEGIES.BINARY
# ---------------------------------------------------

PARAM_RANGE = (8, 104, 8)
RESOLUTION = 1950

#RESULTS_FOLDER = os.path.join("experiments", "results_hyperparam", DATASET_NAME, DBM_TECHNIQUE, PROJECTION, f"{FAST_DECODING_STRATEGY.value}")
RESULTS_FOLDER = os.path.join("experiments", "results_hyperparam", DATASET_NAME, DBM_TECHNIQUE, SDBM_TECHNIQUE, f"{FAST_DECODING_STRATEGY.value}")  # the results folder for SDBM
IMG_SUBFOLDER = os.path.join(RESULTS_FOLDER, "img")
EXPERIMENT_RESULTS_PATH = os.path.join(RESULTS_FOLDER, "experiment_results.txt")
ERRORS_RESULTS_PATH = os.path.join(RESULTS_FOLDER, "label_errors.txt")

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

if not os.path.exists(IMG_SUBFOLDER):
    os.makedirs(IMG_SUBFOLDER)

def hyperparam_run_times():
    if len(os.listdir(IMG_SUBFOLDER)) != 0:
        print("WARNING: The experiment was already run. If you want to run it again, please delete the folder: ", RESULTS_FOLDER)
        print("WARNING: Skipping the experiment...")
        return
    # ---------------------------------------------------

    # Prepare the data
    X_train, X_test, Y_train, Y_test = import_data(dataset_name=DATASET_NAME)

    # Prepare the DBM
    dbm = import_dbm(dbm_technique=DBM_TECHNIQUE,
                     classifier_path=CLASSIFIER_PATH)

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
                                  nn_architecture=SDBM_TECHNIQUE,
                                  load_folder=LOAD_FOLDER,
                                  resolution=10
                                  )

    DECODER = {
        FAST_DBM_STRATEGIES.BINARY: dbm._get_img_dbm_fast_,
        FAST_DBM_STRATEGIES.CONFIDENCE_BASED: dbm._get_img_dbm_fast_confidences_strategy,
        FAST_DBM_STRATEGIES.CONFIDENCE_INTERPOLATION: dbm._get_img_dbm_fast_confidence_interpolation_strategy,
    }

    with open(EXPERIMENT_RESULTS_PATH, "w") as f:
        f.write("B,TIME\n")
    with open(ERRORS_RESULTS_PATH, "w") as f:
            f.write("B,ERROR,ERROR RATE\n")

    # ---------------------------------------------------
    # ---------------------------------------------------
    # Run the generation of the boundary map for different resolutions
    for B in range(*PARAM_RANGE):
        start = time.time()
        if FAST_DECODING_STRATEGY != FAST_DBM_STRATEGIES.NONE:
            img, _, _ = DECODER[FAST_DECODING_STRATEGY](RESOLUTION, initial_resolution=B)
        else:
            print("WARNING: the fast decoding strategy NONE is not supported in this experiment")
            return
        end = time.time()
        decoding_time = round(end - start, 3)
        print("Parameter B: ", B, "Decoding time: ", decoding_time)
        # ---------------------------------------------------
        # Save the results
        with open(EXPERIMENT_RESULTS_PATH, "a") as f:
            f.write(str(B) + "," + str(decoding_time) + "\n")

        img_path = os.path.join(IMG_SUBFOLDER, str(B) + ".npy")
        save_result(img_path, img)
        
        # compute the labels error
        ground_truth_img_folder = os.path.join(RESULTS_FOLDER.replace("results_hyperparam", "results").replace(FAST_DECODING_STRATEGY.value, "none"), "img")
        ground_truth_img_path = os.path.join(ground_truth_img_folder, str(RESOLUTION) + ".npy")
        with open(ground_truth_img_path, "rb") as f:
            ground_truth_img = np.load(f)
        
        # computing the labels error
        error, error_rate = compute_error(img, ground_truth_img)
        line = f"{B},{str(error)},{str(error_rate)}\n"
        with open(ERRORS_RESULTS_PATH, "a") as file:
            file.write(line)
        
def plot_hyperparameter_runtimes():
    path = os.path.join(EXPERIMENT_RESULTS_PATH)
    df = pd.read_csv(path)
    plt.clf()
    plt.plot(df["B"], df["TIME"], label=FAST_DECODING_STRATEGY.value)
    
    plt.title("Run times")
    plt.xlabel("B")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, "resolutions_experiment.png"))
    #plt.show()
    

def plot_hyperparameter_label_errors():
    path = os.path.join(ERRORS_RESULTS_PATH)
    df = pd.read_csv(path)
    plt.clf()
    plt.plot(df["B"], df["ERROR RATE"], label=FAST_DECODING_STRATEGY.value)

    plt.title("Run times")
    plt.xlabel("B")
    plt.ylabel("Labels error rate (%)")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, "label_errors_experiment.png"))
    #plt.show()
