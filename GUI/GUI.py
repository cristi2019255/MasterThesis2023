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
from shutil import rmtree

TITLE = "Classifiers visualization tool"
WINDOW_SIZE = (1300, 800)
BACKGROUND_COLOR = "#252526"
BUTTON_PRIMARY_COLOR = "#007acc"
TEXT_COLOR = "#ffffff"
RIGHTS_MESSAGE = "© 2023 Cristian Grosu. All rights reserved."
RIGHTS_MESSAGE_2 = "Made by Cristian Grosu for Utrecht University Master Thesis in 2023"

DEFAULT_DBM_IMAGE_PATH = os.path.join(os.getcwd(), "results", "MNIST", "2D_boundary_mapping.png")
TMP_FOLDER = os.path.join(os.getcwd(), "tmp")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable tensorflow logs

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

import PySimpleGUI as sg
from GUI.LoggerGUI import LoggerGUI
from utils.DBM import DBM
from utils.Logger import Logger
from PIL import Image, ImageTk
from GUI.DBMPlotter import DBMPlotter

# TODO: delete this after refactoring
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model

from utils.reader import import_mnist_dataset

class GUI:
    def __init__(self):
        self.create_tmp_folder()
        self.window = self.build()
        self.logger = Logger(name = "GUI")
        
        # --------------- Data ---------------
        self.upload_data()
        
        # --------------- Classifier ---------------
        self.upload_classifier()
        # --------------- DBM ---------------
        self.dbm_plotter = None
    
    def create_tmp_folder(self):
        if not os.path.exists(TMP_FOLDER):
            os.makedirs(TMP_FOLDER)
    
    def remove_tmp_folder(self):
        if os.path.exists(TMP_FOLDER):
            rmtree(TMP_FOLDER)
    
    def upload_data(self):
        # import MNIST dataset
        (X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
        
        # limiting to first 5000 samples for testing
        SAMPLES_LIMIT = 5000
        X_train = X_train[:int(0.7 * SAMPLES_LIMIT)]
        Y_train = Y_train[:int(0.7 * SAMPLES_LIMIT)]
        X_test = X_test[:int(0.3 * SAMPLES_LIMIT)]
        Y_test = Y_test[:int(0.3 * SAMPLES_LIMIT)]
        
        
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        
        
        num_classes = np.unique(Y_train).shape[0]
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.num_classes = num_classes
        
    def upload_classifier(self):
        self.classifier = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes, activation=tf.nn.softmax)
        ])
        self.classifier.build(input_shape=(None, 28, 28))
        file = os.path.join(TMP_FOLDER, "classifier.png")
        plot_model(self.classifier, to_file=file, show_shapes=True, show_layer_names=True)    
        img = Image.open(file)
        img.thumbnail((300, 300), Image.ANTIALIAS)
        # Convert im to ImageTk.PhotoImage after window finalized
        image = ImageTk.PhotoImage(image=img)
        self.window["-CLASSIFIER IMAGE-"].update(data=image)
        self.window.refresh()
    
    def build(self):
        layout = self._get_layout()
        window = sg.Window(TITLE, layout, size=WINDOW_SIZE, background_color=BACKGROUND_COLOR)
        window.finalize()
        return window
        
    def start(self):
        while True:
            event, values = self.window.read()
            
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
        
            self.handle_event(event, values)
        
        self.stop()
    
    def stop(self):
        self.logger.log("Removing tmp folder...")
        self.remove_tmp_folder()
        self.logger.log("Closing the application...")
        self.window.close()
    
    def handle_event(self, event, values):        
        # Folder name was filled in, make a list of files in the folder
        EVENTS = {
            "-FOLDER-": self.handle_select_folder_event,
            "-FILE LIST-": self.handle_file_list_event,
            "-DBM BTN-": self.handle_get_decision_boundary_mapping_event,
            "-DBM TECHNIQUE-": self.handle_change_dbm_technique_event,
            "-DBM IMAGE-": self.handle_dbm_image_event,
        }
        
        EVENTS[event](event, values)
        
    
    def _get_layout(self):        
        data_dir = os.getcwd()

        dbm_techniques = [
            "Autoencoder",
        ]
        
        data_files_list_column = [
            [
                sg.Text(size= (15, 1), text = "Data Folder", background_color=BACKGROUND_COLOR),
                sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
                sg.FolderBrowse(button_text="Browse folder", button_color=(TEXT_COLOR, BUTTON_PRIMARY_COLOR), initial_folder=data_dir, size = (22,1)),
            ],
            [
               sg.Text("Choose the data file from the list: ", background_color=BACKGROUND_COLOR),
            ],
            [   
               sg.Text(key="-TOUT-", background_color=BACKGROUND_COLOR, size=(70, 1)),            
            ],
            [
                sg.Listbox(
                    values=[], enable_events=True, size=(70, 30), key="-FILE LIST-"
                )
            ],
            
            [
                sg.Button("Show the Decision Boundary Mapping", button_color=(TEXT_COLOR, BUTTON_PRIMARY_COLOR), size = (69,1), key = "-DBM BTN-"),
            ],
            [
                sg.Text("", size=(10,1), background_color=BACKGROUND_COLOR)    
            ],
            [
                sg.Text( RIGHTS_MESSAGE, background_color=BACKGROUND_COLOR),
            ],
            [
                sg.Text( RIGHTS_MESSAGE_2, background_color=BACKGROUND_COLOR),
            ]
        ]
        
        results_column = [
            [
                sg.Text("Which dbm technique would you like to use?", background_color=BACKGROUND_COLOR, size=(60,1)),
                sg.Combo(
                    values=dbm_techniques,
                    default_value=dbm_techniques[0],
                    size=(20, 1),
                    key="-DBM TECHNIQUE-",
                    enable_events=True,
                ),
            ],
            # ---------------------------------------------------------------------------------------------------
            [
               sg.Column([
                    [sg.Text("Classifier: ", background_color=BACKGROUND_COLOR)],
                    [sg.Image(key="-CLASSIFIER IMAGE-", size=(30, 30), visible=True, enable_events=True)],   
                ], background_color=BACKGROUND_COLOR),
            ],
            [
               sg.Column([
                    [sg.Text("Decision boundary map: ", background_color=BACKGROUND_COLOR)],
                    [sg.Text("Loading... ", background_color=BACKGROUND_COLOR, visible=False, key="-DBM IMAGE LOADING-")],
                    [sg.Image(key="-DBM IMAGE-", size=(80, 80), visible=False, enable_events=True)],   
                ], background_color=BACKGROUND_COLOR),
            ],
            [
                sg.HSeparator(),
            ],
            [
                sg.Column([
                    [sg.Text("Logger: ", background_color=BACKGROUND_COLOR)],
                    [sg.Multiline("",size=(80, 20), key="-LOGGER-", background_color=TEXT_COLOR, text_color=BACKGROUND_COLOR, expand_y=True, auto_size_text=True)],   
                ], background_color=BACKGROUND_COLOR),    
            ]
        ]
        
        
        layout = [
            [
                sg.Column(data_files_list_column, background_color=BACKGROUND_COLOR),
                sg.VSeparator(),
                sg.Column(results_column, background_color=BACKGROUND_COLOR),
            ],
        ]

        return layout


    def handle_select_folder_event(self, event, values):
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [ f for f in file_list
                   if os.path.isfile(os.path.join(folder, f))
                ]
        self.window["-FILE LIST-"].update(fnames)
        
    
    def handle_file_list_event(self, event, values):
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                self.window["-TOUT-"].update(filename)
            except Exception as e:
                self.logger.error("Error while loading data file" + str(e))
    
    def switch_visibility(self, elements, visible):
        for x in elements:
            self.window[x].update(visible=visible)
        self.window.refresh()
        
    def handle_change_dbm_technique_event(self, event, values):
        self.logger.log("Changed dbm technique to: " + values["-DBM TECHNIQUE-"])
        
    def handle_dbm_image_event(self, event, values):
        self.logger.log("Clicked on the dbm image")
        if self.dbm_plotter is None:
            self.logger.warn("No image to show")
            return
        self.dbm_plotter.initialize_plot()
        self.dbm_plotter.plot()
        
    def handle_get_decision_boundary_mapping_event(self, event, values):
        # update loading state
        self.switch_visibility(["-DBM IMAGE LOADING-"], True)
            
        dbm_logger = LoggerGUI(name = "Decision Boundary Mapper", output = self.window["-LOGGER-"], update_callback = self.window.refresh)
        dbm = DBM(classifier = self.classifier, logger = dbm_logger)
        img, encoded_training_data, encoded_testing_data = dbm.generate_boundary_map(
                                self.X_train, 
                                self.Y_train, 
                                self.X_test, 
                                self.Y_test, 
                                train_epochs=10, 
                                train_batch_size=128,
                                resolution=256,
                                save_file_path=DEFAULT_DBM_IMAGE_PATH,
                                show_autoencoder_predictions=False,
                                show_encoded_corpus=False,
                                )

        # ---------------------------------
        # update the GUI dbm attributes
        self.dbm_plotter = DBMPlotter(img = img, 
                                      num_classes = self.num_classes, 
                                      encoded_train = encoded_training_data, 
                                      encoded_test = encoded_testing_data,
                                      X_train = self.X_train,
                                      Y_train = self.Y_train,
                                      X_test = self.X_test,
                                      Y_test = self.Y_test)
        
        # ---------------------------------
        # update the dbm image
        img = Image.fromarray(np.uint8(self.dbm_plotter.color_img*255))
        img.thumbnail((700, 700), Image.ANTIALIAS)
        # Convert im to ImageTk.PhotoImage after window finalized
        image = ImageTk.PhotoImage(image=img)
        
        self.window["-DBM IMAGE-"].update(data=image)
        
        self.switch_visibility(["-DBM IMAGE LOADING-"], False)
        self.switch_visibility(["-DBM IMAGE-"], True)