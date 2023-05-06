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
import json
from shutil import rmtree
import tensorflow as tf
import numpy as np

from .DBMPlotterGUI import DBMPlotterGUI
from ..DBM import SDBM, DBM, NNArchitecture
from ..Logger import LoggerGUI, Logger
from ..utils import import_csv_dataset, import_mnist_dataset, import_cifar10_dataset, import_fashion_mnist_dataset


DBM_NNINV_TECHNIQUE = "nnInv"
SDBM_SSNP_TECHNIQUE = "ssnp"
SDBM_AUTOENCODER_TECHNIQUE = "autoencoder"

DBM_TECHNIQUES = {
    SDBM_AUTOENCODER_TECHNIQUE: SDBM,
    SDBM_SSNP_TECHNIQUE: SDBM,
    DBM_NNINV_TECHNIQUE: DBM,
}

PROJECTION_TECHNIQUES = [
    "t-SNE",
    "UMAP",
    "PCA",
]

DATASETS_IMPORTERS = {
    "MNIST": import_mnist_dataset,
    "FASHION MNIST": import_fashion_mnist_dataset,
    "CIFAR10": import_cifar10_dataset,
}
TRAIN_2D_FILE_NAME = "train_2d.npy"
TEST_2D_FILE_NAME = "test_2d.npy"
DBM_FOLDER_NAME = "DBM"
SDBM_FOLDER_NAME = "SDBM"
HISTORY_FILE_NAME = "history.json"

TMP_FOLDER = os.path.join(os.getcwd(), "tmp")
SAMPLES_LIMIT = 5000  # Limit the number of samples to be loaded from the dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable tensorflow logs
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""


class GUIController:
    def __init__(self, window, gui):
        self.window = window
        self.gui = gui
        self.create_tmp_folder()
        self.logger = Logger(name="GUIController")
        self.dbm_logger = LoggerGUI(name="DBM logger", output=self.window["-LOGGER-"], update_callback=self.window.refresh)

        
        # --------------- Data set ----------------
        self.dataset_name = "Dataset"
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None

        # --------------- Classifier --------------
        self.classifier = None
       
    def stop(self):
        self.logger.log("Clearing resources...")
        # self.logger.log("Removing tmp folder...")
        # self.remove_tmp_folder()
        
    def create_tmp_folder(self):
        if not os.path.exists(TMP_FOLDER):
            os.makedirs(TMP_FOLDER)

    def remove_tmp_folder(self):
        if os.path.exists(TMP_FOLDER):
            rmtree(TMP_FOLDER)

    def switch_visibility(self, elements, visible):
        for x in elements:
            self.window[x].update(visible=visible)
        self.window.refresh()

    def handle_select_classifier_folder_event(self, event, values):
        folder = values["-CLASSIFIER FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [f for f in file_list
                  if (os.path.isfile(os.path.join(folder, f)) and (f.lower().endswith((".h5")))) or os.path.isdir(os.path.join(folder, f))]

        self.window["-CLASSIFIER FILE LIST-"].update(fnames)

    def handle_select_data_folder_event(self, event, values):
        folder = values["-DATA FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [f for f in file_list
                  if os.path.isfile(os.path.join(folder, f)) and (f.lower().endswith((".csv")) or f.lower().endswith((".txt")))
                  ]
        self.window["-DATA FILE LIST-"].update(fnames)

    def handle_file_list_event(self, event, values):
        try:
            filename = os.path.join(
                values["-DATA FOLDER-"], values["-DATA FILE LIST-"][0])
            self.window["-DATA FILE TOUT-"].update(filename)
        except Exception as e:
            self.logger.error("Error while loading data file" + str(e))

    def handle_classifier_file_list_event(self, event, values):
        try:
            filename = os.path.join(
                values["-CLASSIFIER FOLDER-"], values["-CLASSIFIER FILE LIST-"][0])
            self.window["-CLASSIFIER PATH TOUT-"].update(filename)
        except Exception as e:
            self.logger.error("Error while loading data file" + str(e))
        
    def handle_upload_classifier_event(self, event, values):
        try:
            fname = os.path.join(
                values["-CLASSIFIER FOLDER-"], values["-CLASSIFIER FILE LIST-"][0])
            self.classifier = tf.keras.models.load_model(fname)
            self.window["-CLASSIFIER PATH TOUT-"].update(fname)
            self.logger.log("Classifier loaded successfully")
            self.dbm_logger.log("Classifier loaded successfully")
            if self.X_train is not None and self.Y_train is not None and self.X_test is not None and self.Y_test is not None:
                self.switch_visibility(["-DBM BTN-"], True)
        except Exception as e:
            self.logger.error("Error while loading classifier" + str(e))
    
    def handle_dbm_technique_event(self, event, values):
        dbm_technique = values["-DBM TECHNIQUE-"]
        self.logger.log(f"DBM technique: {dbm_technique}")
        
        if dbm_technique == DBM_NNINV_TECHNIQUE:
            self.switch_visibility(["-PROJECTION TECHNIQUE TEXT-", "-PROJECTION TECHNIQUE-"], True)
            return
        
        self.switch_visibility(["-PROJECTION TECHNIQUE TEXT-", "-PROJECTION TECHNIQUE-"], False)
        
    def handle_projection_technique_event(self, event, values):
        projection_technique = values["-PROJECTION TECHNIQUE-"]
        self.logger.log(f"Projection technique: {projection_technique}")
        
    def handle_upload_train_data_event(self, event, values):
        try:
            filename = os.path.join(values["-DATA FOLDER-"], values["-DATA FILE LIST-"][0])
            self.X_train, self.Y_train = import_csv_dataset(filename, limit=int(0.7*SAMPLES_LIMIT))

            self.num_classes = np.unique(self.Y_train).shape[0]
            if self.classifier is not None and self.X_test is not None and self.Y_test is not None:
                self.switch_visibility(["-DBM BTN-"], True)

            self.window["-TRAIN DATA FILE-"].update("Training data file: " + filename)
            self.window["-TRAIN DATA SHAPE-"].update(f"Training data shape: X {self.X_train.shape} Y {self.Y_train.shape}")
        except Exception as e:
            self.dbm_logger.error("Error while loading data file" + str(e))

    def handle_upload_test_data_event(self, event, values):
        try:
            filename = os.path.join(values["-DATA FOLDER-"], values["-DATA FILE LIST-"][0])
            self.X_test, self.Y_test = import_csv_dataset(filename, limit=int(0.3*SAMPLES_LIMIT))
            if self.classifier is not None and self.X_train is not None and self.Y_train is not None:
                self.switch_visibility(["-DBM BTN-"], True)

            self.window["-TEST DATA FILE-"].update("Testing data file: " + filename)
            self.window["-TEST DATA SHAPE-"].update(f"Testing data shape: X {self.X_test.shape} Y {self.Y_test.shape}")
        except Exception as e:
            self.dbm_logger.error("Error while loading data file" + str(e))



    def handle_upload_known_data_event(self, event, values):
        self.dataset_name = event[len("-UPLOAD "):-len(" DATA BTN-")]
        
        (X_train, Y_train), (X_test, Y_test) = DATASETS_IMPORTERS[self.dataset_name]()

        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        X_train, Y_train = X_train[:int(0.7*SAMPLES_LIMIT)], Y_train[:int(0.7*SAMPLES_LIMIT)]
        X_test, Y_test = X_test[:int(0.3*SAMPLES_LIMIT)], Y_test[:int(0.3*SAMPLES_LIMIT)]

        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test
        self._post_uploading_processing_()
    
    def _post_uploading_processing_(self):
        self.num_classes = np.unique(self.Y_train).shape[0]

        self.window["-TRAIN DATA FILE-"].update(f"Training data: {self.dataset_name}")
        self.window["-TRAIN DATA SHAPE-"].update(f"Training data shape: X {self.X_train.shape} Y {self.Y_train.shape}")

        self.window["-TEST DATA FILE-"].update(f"Testing data: {self.dataset_name}")
        self.window["-TEST DATA SHAPE-"].update(f"Testing data shape: X {self.X_test.shape} Y {self.Y_test.shape}")

        if self.classifier is not None:
            self.switch_visibility(["-DBM BTN-"], True)

    def handle_get_decision_boundary_mapping_event(self, event, values):
        if self.classifier is None:
            self.dbm_logger.error("No classifier provided, impossible to generate the DBM...")
            return

        if self.X_train is None or self.Y_train is None or self.X_test is None or self.Y_test is None:
            self.dbm_logger.error("Data is incomplete impossible to generate the DBM...")
            return

        # update loading state
        self.switch_visibility(["-DBM IMAGE-"], False)
        self.switch_visibility(["-DBM TEXT-", "-DBM IMAGE LOADING-"], True)

        dbm = DBM_TECHNIQUES[values["-DBM TECHNIQUE-"]](classifier=self.classifier, logger=self.dbm_logger)

        projection_technique = values["-PROJECTION TECHNIQUE-"]
        resolution = 256  # values["-DBM IMAGE RESOLUTION INPUT-"]

        self.dbm_logger.log(f"DBM resolution: {resolution}")

        tmp_folder = os.path.join("tmp", self.dataset_name)
        save_folder = tmp_folder
        dbm_technique = values["-DBM TECHNIQUE-"]
        
        if dbm_technique == DBM_NNINV_TECHNIQUE:
            dbm_folder = os.path.join(tmp_folder, DBM_FOLDER_NAME)
            save_folder = os.path.join(dbm_folder, projection_technique)

            if not os.path.exists(os.path.join(save_folder, TRAIN_2D_FILE_NAME)):
                X_train_2d = None
            else:
                with open(os.path.join(save_folder, TRAIN_2D_FILE_NAME), "rb") as f:
                    X_train_2d = np.load(f)
            if not os.path.exists(os.path.join(save_folder, TEST_2D_FILE_NAME)):
                X_test_2d = None
            else:
                with open(os.path.join(save_folder, TEST_2D_FILE_NAME), "rb") as f:
                    X_test_2d = np.load(f)

            dbm_info = dbm.generate_boundary_map(
                Xnd_train=self.X_train,
                Xnd_test=self.X_test,
                X2d_train=X_train_2d,
                X2d_test=X_test_2d,
                resolution=resolution,
                load_folder=dbm_folder,
                projection=projection_technique
            )
        else:
            save_folder = os.path.join(tmp_folder, SDBM_FOLDER_NAME)
            projection_technique = None
            dbm_info = dbm.generate_boundary_map(
                X_train=self.X_train,
                Y_train=self.Y_train,
                X_test=self.X_test,
                Y_test=self.Y_test,
                nn_architecture=NNArchitecture(dbm_technique),
                resolution=resolution,
                load_folder=save_folder,
            )

        img, img_confidence, encoded_training_data, encoded_testing_data = dbm_info
        
        show_dbm_history = values["-DBM HISTORY CHECKBOX-"]
        if show_dbm_history:
            path = os.path.join(save_folder, HISTORY_FILE_NAME)
            training_history = None
            if os.path.exists(path):   
                with open(path, 'r') as f:
                    training_history = json.load(f)
            self.gui.show_dbm_history(training_history)

        # ---------------------------------
        # create a GUI window for Decision Boundary Mapping
        return DBMPlotterGUI(
            dbm_model=dbm,
            img=img,
            img_confidence=img_confidence,
            encoded_train=encoded_training_data,
            encoded_test=encoded_testing_data,
            X_train=self.X_train,
            Y_train=self.Y_train,
            X_test=self.X_test,
            Y_test=self.Y_test,
            main_gui=self.gui,  # reference to the main GUI
            save_folder=save_folder,
            projection_technique=projection_technique,
        )
        