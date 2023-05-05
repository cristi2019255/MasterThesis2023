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
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import numpy as np
from PIL import Image, ImageTk

from ..Logger import Logger
from .GUIController import DBM_TECHNIQUES, PROJECTION_TECHNIQUES, GUIController

sg.theme('DarkBlue1')
TITLE = "Classifiers visualization tool"
WINDOW_SIZE = (1150, 700)
BLACK_COLOR = "#252526"
BUTTON_PRIMARY_COLOR = "#007acc"
WHITE_COLOR = "#ffffff"
RIGHTS_MESSAGE = "© 2023 Cristian Grosu. All rights reserved."
RIGHTS_MESSAGE_2 = "Made by Cristian Grosu for Utrecht University Master Thesis in 2023"
APP_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "main_icon.png")


class GUI:
    def __init__(self):
        self.window = self.build_window()
        self.controller = GUIController(self.window, self)
        self.logger = Logger(name="GUI")

        # --------------- DBM ---------------
        self.dbm_plotter_gui = None
        

    def build_window(self):
        window = sg.Window(TITLE,
                           layout=self._get_layout(),
                           size=WINDOW_SIZE,
                           icon=APP_ICON_PATH,
                           resizable=False,
                           )
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
        self.controller.stop()
        self.logger.log("Closing the application...")
        self.window.close()

    def _get_layout(self):
        data_files_list_column = [
            [
                sg.Text(text="Data Folder"),
                sg.In(enable_events=True, key="-DATA FOLDER-",
                      background_color=WHITE_COLOR, text_color=BLACK_COLOR, expand_x=True),
                sg.FolderBrowse(button_text="Browse folder", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), initial_folder=os.getcwd()),
            ],
            [
                sg.Text("Choose the data file from the list: ", expand_x=True),
            ],
            [
                sg.Text(key="-DATA FILE TOUT-", expand_x=True),
            ],
            [
                sg.Listbox(
                    values=[], enable_events=True, key="-DATA FILE LIST-", background_color=WHITE_COLOR, text_color=BLACK_COLOR, expand_x=True, expand_y=True
                )
            ],
            [
                sg.Button("Upload train data for DBM", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), expand_x=True, key="-UPLOAD TRAIN DATA BTN-"),
                sg.Button("Upload test data for DBM", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), expand_x=True, key="-UPLOAD TEST DATA BTN-"),
            ],
            [
                sg.Text("Training data file: ",
                        key="-TRAIN DATA FILE-",  expand_x=True),
            ],
            [
                sg.Text("Training data shape: ",
                        key="-TRAIN DATA SHAPE-", expand_x=True),
            ],
            [
                sg.Text("Testing data file: ",
                        key="-TEST DATA FILE-", expand_x=True),
            ],
            [
                sg.Text("Testing data shape: ",
                        key="-TEST DATA SHAPE-", expand_x=True),
            ],
            [
                sg.Button("Upload MNIST Data set", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), expand_x=True, key="-UPLOAD MNIST DATA BTN-"),
            ],
            [
                sg.Button("Upload FASHION MNIST Data set", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), expand_x=True, key="-UPLOAD FASHION MNIST DATA BTN-"),
            ],
            [
                sg.Button("Upload CIFAR10 Data set", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), expand_x=True, key="-UPLOAD CIFAR10 DATA BTN-"),
            ],
            [
                sg.Text(text="Classifier Folder"),
                sg.In(enable_events=True, key="-CLASSIFIER FOLDER-",
                      background_color=WHITE_COLOR, text_color=BLACK_COLOR, expand_x=True),
                sg.FolderBrowse(button_text="Browse folder", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), initial_folder=os.getcwd()),
            ],
            [
                sg.Text(
                    "Choose the classifier file(.h5) or folder from the list: ", expand_x=True),
            ],
            [
                sg.Text(key="-CLASSIFIER PATH TOUT-", expand_x=True),
            ],
            [
                sg.Listbox(
                    values=[], enable_events=True, key="-CLASSIFIER FILE LIST-", background_color=WHITE_COLOR, text_color=BLACK_COLOR, expand_x=True, expand_y=True
                )
            ],
            [
                sg.Button("Upload classifier", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), expand_x=True, visible=True, key="-UPLOAD CLASSIFIER-"),
            ],
            [
                sg.Text("", expand_x=True)
            ],
            [
                sg.Text(RIGHTS_MESSAGE, expand_x=True),
            ],
            [
                sg.Text(RIGHTS_MESSAGE_2, expand_x=True),
            ]
        ]

        results_column = [
            [
                sg.Text("Which dbm technique would you like to use?",
                        size=(45, 1)),
                sg.Combo(
                    values=list(DBM_TECHNIQUES.keys()),
                    default_value=list(DBM_TECHNIQUES.keys())[0],
                    expand_x=True,
                    key="-DBM TECHNIQUE-",
                    enable_events=True,
                    background_color=WHITE_COLOR, text_color=BLACK_COLOR,
                ),
            ],
            [
                sg.Text("Which Projection technique would you like to use?", size=(
                    45, 1), key="-PROJECTION TECHNIQUE TEXT-", visible=False),
                sg.Combo(
                    values=PROJECTION_TECHNIQUES,
                    default_value=PROJECTION_TECHNIQUES[0],
                    expand_x=True,
                    key="-PROJECTION TECHNIQUE-",
                    enable_events=True,
                    visible=False,
                    background_color=WHITE_COLOR, text_color=BLACK_COLOR,
                ),
            ],
            [
                sg.Text("Show Decision Boundary Mapper NN history: ",
                        expand_x=True, key="-DBM HISTORY TEXT-", visible=True),
                sg.Checkbox("", default=False,
                            key="-DBM HISTORY CHECKBOX-", visible=True),
            ],
            # [
            #    sg.Text("Image resolution of the Decision Boundary Mapper: ", size=(45,1), expand_x=True, key="-DBM IMAGE RESOLUTION TEXT-", visible=True),
            #    sg.InputText("256", key="-DBM IMAGE RESOLUTION INPUT-", visible=True, background_color=WHITE_COLOR, text_color=BLACK_COLOR),
            # ],
            [
                sg.Button("Show the Decision Boundary Mapping", button_color=(
                    WHITE_COLOR, BUTTON_PRIMARY_COLOR), expand_x=True, visible=False, key="-DBM BTN-"),
            ],
            # ---------------------------------------------------------------------------------------------------
            [sg.Text("Decision boundary map: ", visible=False,
                     expand_x=True, key="-DBM TEXT-", justification='center')],
            [sg.Text("Loading... ",  visible=False, expand_x=True,
                     key="-DBM IMAGE LOADING-", justification='center')],
            [sg.Image(key="-DBM IMAGE-", visible=False, expand_x=True,
                      expand_y=True, enable_events=True)],
            [
                sg.HSeparator(),
            ],
            [
                sg.Column([
                    [sg.Text("Logger: ")],
                    [sg.Multiline("", expand_x=True, expand_y=True, key="-LOGGER-",
                                  background_color=WHITE_COLOR, text_color=BLACK_COLOR, auto_size_text=True)],
                ], expand_x=True, expand_y=True),
            ]
        ]

        layout = [
            [
                sg.Column(data_files_list_column,
                          expand_x=True, expand_y=True),
                sg.VSeparator(),
                sg.Column(results_column, expand_x=True, expand_y=True),
            ],
        ]

        return layout

    def switch_visibility(self, elements, visible):
        for x in elements:
            self.window[x].update(visible=visible)
        self.window.refresh()

    def handle_event(self, event, values):
        EVENTS = {
            "-DATA FOLDER-": self.controller.handle_select_data_folder_event,
            "-CLASSIFIER FOLDER-": self.controller.handle_select_classifier_folder_event,
            "-DATA FILE LIST-": self.controller.handle_file_list_event,
            "-CLASSIFIER FILE LIST-": self.controller.handle_classifier_file_list_event,
            "-DBM TECHNIQUE-": self.controller.handle_dbm_technique_event,
            "-DBM BTN-": self.handle_get_decision_boundary_mapping_event,
            "-DBM IMAGE-": self.handle_dbm_image_event,
            "-PROJECTION TECHNIQUE-": self.controller.handle_projection_technique_event,
            "-UPLOAD TRAIN DATA BTN-": self.controller.handle_upload_train_data_event,
            "-UPLOAD TEST DATA BTN-": self.controller.handle_upload_test_data_event,
            "-UPLOAD MNIST DATA BTN-": self.controller.handle_upload_known_data_event,
            "-UPLOAD FASHION MNIST DATA BTN-": self.controller.handle_upload_known_data_event,
            "-UPLOAD CIFAR10 DATA BTN-": self.controller.handle_upload_known_data_event,
            "-UPLOAD CLASSIFIER-": self.controller.handle_upload_classifier_event,
        }

        EVENTS[event](event, values)
    
    def handle_changes_in_dbm_plotter(self):
        # update loading state
        self.switch_visibility(["-DBM IMAGE-"], False)
        self.switch_visibility(["-DBM TEXT-", "-DBM IMAGE LOADING-"], True)

        # ---------------------------------
        # update the dbm image
        img = Image.fromarray(np.uint8(self.dbm_plotter_gui.color_img*255))
        WINDOW_IMAGE_RESOLUTION = 256
        img.thumbnail((WINDOW_IMAGE_RESOLUTION, WINDOW_IMAGE_RESOLUTION), Image.ANTIALIAS)
        # Convert im to ImageTk.PhotoImage after window finalized
        image = ImageTk.PhotoImage(image=img)
        self.window["-DBM IMAGE-"].update(data=image)

        self.switch_visibility(["-DBM IMAGE LOADING-"], False)
        self.switch_visibility(["-DBM IMAGE-"], True)

    def show_dbm_history(self, training_history):
        # this is for plotting the training history
        plt.close()
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_title("Loss")
        ax1.plot(training_history["loss"], label="loss")
        ax1.plot(training_history["val_loss"], label="val_loss")
        ax1.legend()

        ax2.set_title("Decoder Accuracy")
        if "decoder_accuracy" in training_history and "val_decoder_accuracy" in training_history:
            ax2.plot(training_history["decoder_accuracy"], label="decoder_accuracy")
            ax2.plot(training_history["val_decoder_accuracy"], label="val_decoder_accuracy")
        else:
            ax2.plot(training_history["accuracy"], label="accuracy")
            ax2.plot(training_history["val_accuracy"], label="val_accuracy")
            
        ax2.legend()

        plt.show()

    def handle_get_decision_boundary_mapping_event(self, event, values):
        self.dbm_plotter_gui = self.controller.handle_get_decision_boundary_mapping_event(event, values)
        self.handle_changes_in_dbm_plotter()
        
        
    def handle_dbm_image_event(self, event, values):
        self.logger.log("Clicked on the dbm image")
        if self.dbm_plotter_gui is None:
            self.logger.warn("Nothing to show")
            return
        self.dbm_plotter_gui.start()
