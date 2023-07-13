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

from ..Logger import Logger, LoggerGUI
from .GUIController import DBM_TECHNIQUES, PROJECTION_TECHNIQUES, DBM_NNINV_TECHNIQUE, CUSTOM_PROJECTION_TECHNIQUE, GUIController
from ..utils import BLACK_COLOR, WHITE_COLOR, RIGHTS_MESSAGE_1, RIGHTS_MESSAGE_2, APP_PRIMARY_COLOR, APP_FONT, APP_FONT_BOLD
from .utils import Collapsible

sg.theme('DarkBlue1')
TITLE = "Classifiers visualization tool"
WINDOW_SIZE = (1150, 700)
APP_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "main_icon.png")
class GUI:
    def __init__(self):
        self.window = self.build_window()
        self.gui_logger = LoggerGUI(name="DBM logger", output=self.window["-LOGGER-"], update_callback=self.window.refresh)
        self.logger = Logger(name="GUI")
        self.controller = GUIController(self.window, self, self.gui_logger)

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
                sg.Button("Upload MNIST Data set", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, key="-UPLOAD MNIST DATA BTN-"),
            ],
            [
                sg.Button("Upload FASHION MNIST Data set", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, key="-UPLOAD FASHION MNIST DATA BTN-"),
            ],
            [
                sg.Button("Upload CIFAR10 Data set", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, key="-UPLOAD CIFAR10 DATA BTN-"),
            ],
            [ sg.HSeparator() ],
            [
                Collapsible([
                    [
                        sg.Text(text="Choose the data folder: ", font=APP_FONT),
                        sg.In(enable_events=True, key="-DATA FOLDER-", visible=False),
                        sg.Button("Browse folder", 
                          button_type=sg.BUTTON_TYPE_BROWSE_FOLDER, 
                          target=(sg.ThisRow, -1), 
                          button_color=(WHITE_COLOR, APP_PRIMARY_COLOR),
                          font=APP_FONT,
                          initial_folder=os.getcwd(),
                          expand_x=True,
                        ), 
                    ],
                    [
                        sg.Text("Choose the data file from the list: ", font=APP_FONT, expand_x=True),
                    ],
                    [
                       sg.pin(sg.Column([
                            [sg.Text(key="-DATA FOLDER TOUT-", font=APP_FONT, expand_x=True)],
                           ], key="-DATA FOLDER TOUT PIN-", visible=False, expand_x=True), 
                        shrink=True, expand_x=True), 
                    ],
                    [
                       sg.pin(sg.Column([
                            [sg.Text(key="-DATA FILE TOUT-", font=APP_FONT, expand_x=True)],
                           ], key="-DATA FILE TOUT PIN-", visible=False, expand_x=True), 
                        shrink=True, expand_x=True), 
                    ],
                    [
                        sg.Listbox(
                            values=[], enable_events=True, key="-DATA FILE LIST-", 
                            background_color=WHITE_COLOR, text_color=BLACK_COLOR, sbar_background_color=APP_PRIMARY_COLOR,
                            expand_x=True, size=(10, 10), font=APP_FONT_BOLD,
                        )
                    ],
                    [
                        sg.Button("Upload train data for DBM", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, key="-UPLOAD TRAIN DATA BTN-"),
                        sg.Button("Upload test data for DBM", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, key="-UPLOAD TEST DATA BTN-"),
                    ],
                    [
                        sg.Button("Upload Data set as folder", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, key="-UPLOAD FOLDER DATASET BTN-"),
                    ],
            
                ], "-UPLOAD DATA COLLAPSABLE-", "Upload the dataset from a folder", collapsed=True, visible=True), 
            ],
            [ sg.HSeparator() ],
            [
                sg.Text("Training data file: ", font=APP_FONT, key="-TRAIN DATA FILE-",  expand_x=True),
            ],
            [
                sg.Text("Training data shape: ", font=APP_FONT, key="-TRAIN DATA SHAPE-", expand_x=True),
            ],
            [
                sg.Text("Testing data file: ", font=APP_FONT, key="-TEST DATA FILE-", expand_x=True),
            ],
            [
                sg.Text("Testing data shape: ", font=APP_FONT, key="-TEST DATA SHAPE-", expand_x=True),
            ],
            [ sg.HSeparator()],
            [
                sg.Text(text="Choose the classifier folder", font=APP_FONT),
                sg.In(enable_events=True, key="-CLASSIFIER FOLDER-", visible=False),
                sg.Button("Browse folder", 
                          button_type=sg.BUTTON_TYPE_BROWSE_FOLDER, 
                          target=(sg.ThisRow, -1), 
                          button_color=(WHITE_COLOR, APP_PRIMARY_COLOR),
                          font=APP_FONT,
                          initial_folder=os.getcwd(),
                          expand_x=True,
                    ),
            ],
            [
                sg.Text("Choose the classifier file(.h5) or folder from the list: ", font=APP_FONT, expand_x=True),
            ],
            [
                sg.pin(sg.Column([
                            [sg.Text(key="-CLASSIFIER PATH TOUT-", font=APP_FONT, expand_x=True)],
                           ], key="-CLASSIFIER PATH TOUT PIN-", visible=False, expand_x=True), 
                shrink=True, expand_x=True), 
            ],
            [
                sg.Listbox(
                    values=[], enable_events=True, key="-CLASSIFIER FILE LIST-", 
                    background_color=WHITE_COLOR, text_color=BLACK_COLOR, sbar_background_color=APP_PRIMARY_COLOR,
                    expand_x=True, expand_y=True, font=APP_FONT_BOLD,
                )
            ],
            [
                sg.Button("Upload classifier", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, visible=True, key="-UPLOAD CLASSIFIER-"),
            ],
            [ sg.HSeparator()],
            [
                sg.Text("", expand_x=True)
            ],
            [
                sg.Text(RIGHTS_MESSAGE_1, font=APP_FONT, expand_x=True),
            ],
            [
                sg.Text(RIGHTS_MESSAGE_2, font=APP_FONT, expand_x=True),
            ]
        ]

        results_column = [
            [
                sg.Text("Which dbm technique would you like to use?", font=APP_FONT, size=(45, 1)),
                sg.Combo(
                    values=list(DBM_TECHNIQUES.keys()),
                    default_value=list(DBM_TECHNIQUES.keys())[0],
                    expand_x=True,
                    font=APP_FONT,
                    button_background_color=APP_PRIMARY_COLOR,
                    key="-DBM TECHNIQUE-",
                    enable_events=True,
                    background_color=WHITE_COLOR, text_color=BLACK_COLOR,
                ),
            ],
            [
                sg.pin(sg.Column([
                    [
                    sg.Text("Which Projection technique would you like to use?", font=APP_FONT, size=(45, 1), key="-PROJECTION TECHNIQUE TEXT-"),
                    sg.Combo(
                        values=PROJECTION_TECHNIQUES,
                        default_value=PROJECTION_TECHNIQUES[0],
                        expand_x=True,
                        font=APP_FONT,
                        button_background_color=APP_PRIMARY_COLOR,
                        key="-PROJECTION TECHNIQUE-",
                        enable_events=True,
                        background_color=WHITE_COLOR, text_color=BLACK_COLOR,
                    ),
                    ]
                ], expand_x=True, key="-PROJECTION TECHNIQUE COLUMN-", visible=False), shrink=True, expand_x=True)    
            ],
            [
                sg.pin(sg.Column([
                    [
                     Collapsible([
                            [
                                sg.Text(text="Choose the 2D data folder", key="-DATA 2D FOLDER TEXT-", font=APP_FONT),
                                sg.In(enable_events=True, key="-DATA 2D FOLDER-", visible=False),
                                sg.Button("Browse folder", 
                                    button_type=sg.BUTTON_TYPE_BROWSE_FOLDER, 
                                    target=(sg.ThisRow, -1), 
                                    button_color=(WHITE_COLOR, APP_PRIMARY_COLOR),
                                    font=APP_FONT,
                                    initial_folder=os.getcwd(),
                                    expand_x=True,
                                ),
                            ],
                            [
                                sg.Listbox(
                                    values=[], enable_events=True, key="-DATA 2D FILE LIST-", 
                                    background_color=WHITE_COLOR, text_color=BLACK_COLOR, sbar_background_color=APP_PRIMARY_COLOR,
                                    expand_x=True, size=(10, 10), font=APP_FONT_BOLD,
                                )
                            ],   
                            [
                                sg.Button("Upload 2D train data", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, expand_y=False, key="-UPLOAD 2D TRAIN DATA BTN-"),
                                sg.Button("Upload 2D test data", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), font=APP_FONT, expand_x=True, expand_y=False, key="-UPLOAD 2D TEST DATA BTN-"),
                            ],
                        ], "-DATA 2D COLLAPSABLE-", "Select the files with 2D representation of the data", collapsed=True, visible=True),
                    ]
                ], key="-DATA 2D COLUMN-", visible=False, expand_x=True),
                shrink = True, expand_x = True)
            ],
            [
                sg.Text("Show Decision Boundary Mapper NN history: ", font=APP_FONT, expand_x=True, key="-DBM HISTORY TEXT-", visible=True),
                sg.Checkbox("", default=False, font=APP_FONT, key="-DBM HISTORY CHECKBOX-", visible=True),
            ],
            # [
            #    sg.Text("Image resolution of the Decision Boundary Mapper: ", size=(45,1), expand_x=True, key="-DBM IMAGE RESOLUTION TEXT-", visible=True),
            #    sg.InputText("256", key="-DBM IMAGE RESOLUTION INPUT-", visible=True, background_color=WHITE_COLOR, text_color=BLACK_COLOR),
            # ],
            [
                sg.Button("Show the Decision Boundary Mapping", font=APP_FONT, button_color=(WHITE_COLOR, APP_PRIMARY_COLOR), expand_x=True, visible=False, key="-DBM BTN-"),
            ],
            # ---------------------------------------------------------------------------------------------------
            [ sg.pin(sg.Column([
                [sg.Text("Decision boundary map: ", font=APP_FONT, expand_x=True, justification='center')],
                ], key="-DBM TEXT-", visible=False, expand_x=True, expand_y=False), 
                shrink=True, expand_x=True),
            ],
            [
                sg.pin(sg.Column([
                      [sg.Text("Loading... ", font=APP_FONT, expand_x=True, justification='center')],
                    ], key="-DBM IMAGE LOADING-", visible=False, expand_x=True, expand_y=False), 
                    shrink=True, expand_x=True),
            ],
            [sg.Image(key="-DBM IMAGE-", visible=False, expand_x=True, expand_y=False, enable_events=True)],
            [
                sg.HSeparator(),
            ],
            [
                sg.Column([
                    [sg.Text("Logger: ", font=APP_FONT,)],
                    [sg.Multiline("", font=APP_FONT, 
                                  expand_x=True, expand_y=True, key="-LOGGER-", 
                                  background_color=WHITE_COLOR, 
                                  text_color=BLACK_COLOR,
                                  sbar_background_color=APP_PRIMARY_COLOR,   
                                  auto_size_text=True)],
                ], expand_x=True, expand_y=True, key="-LOGGER COLUMN-"),
            ]
        ]

        layout = [
            [
                sg.Column(data_files_list_column, expand_x=True, expand_y=True),
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
            "-DATA FOLDER-": self.handle_select_data_folder_event,
            "-DATA 2D FOLDER-": self.handle_select_2d_data_folder_event,
            "-CLASSIFIER FOLDER-": self.handle_select_classifier_folder_event,
        
            "-DATA FILE LIST-": self.controller.handle_file_list_event,
            "-DATA 2D FILE LIST-": self.handle_2d_file_list_event,
          
            
            "-CLASSIFIER FILE LIST-": self.controller.handle_classifier_file_list_event,
            
            "-DBM TECHNIQUE-": self.handle_dbm_technique_event,
            "-DBM BTN-": self.handle_get_decision_boundary_mapping_event,
            "-DBM IMAGE-": self.handle_dbm_image_event,
            "-PROJECTION TECHNIQUE-": self.handle_projection_technique_event,
            
            "-UPLOAD 2D TRAIN DATA BTN-": self.controller.handle_upload_2d_data_event,
            "-UPLOAD 2D TEST DATA BTN-": self.controller.handle_upload_2d_data_event,
            "-UPLOAD TRAIN DATA BTN-": self.controller.handle_upload_train_data_event,
            "-UPLOAD TEST DATA BTN-": self.controller.handle_upload_test_data_event,
            "-UPLOAD MNIST DATA BTN-": self.controller.handle_upload_known_data_event,
            "-UPLOAD FASHION MNIST DATA BTN-": self.controller.handle_upload_known_data_event,
            "-UPLOAD CIFAR10 DATA BTN-": self.controller.handle_upload_known_data_event,
            "-UPLOAD FOLDER DATASET BTN-": self.controller.handle_upload_folder_dataset_event,
            
            "-UPLOAD CLASSIFIER-": self.controller.handle_upload_classifier_event,
            
            # collapsable components
            "-DATA 2D COLLAPSABLE-/-BUTTON-": self.handle_collapse_event,
            "-DATA 2D COLLAPSABLE-/-TITLE-": self.handle_collapse_event,
            "-UPLOAD DATA COLLAPSABLE-/-BUTTON-": self.handle_collapse_event,
            "-UPLOAD DATA COLLAPSABLE-/-TITLE-": self.handle_collapse_event,
        }

        EVENTS[event](event, values)
    
    def handle_collapse_event(self, event, values):
        collapsable_key = event.split('/')[0]
        component = self.window[collapsable_key]
        btn_component = self.window[collapsable_key + "/" +"-BUTTON-"]
        component.update(visible=not component.visible)
        btn_component.update(component.metadata[0] if component.visible else component.metadata[1])

    def handle_2d_file_list_event(self, event, values):
        filename = os.path.join(values["-DATA 2D FOLDER-"], values["-DATA 2D FILE LIST-"][0])
        self.gui_logger.log(f"Selected file: {filename}")
        
    def handle_select_data_folder_event(self, event, values):
        folder = values["-DATA FOLDER-"]
        self.window["-DATA FOLDER TOUT-"].update(folder)
        self.switch_visibility(["-DATA FOLDER TOUT PIN-"], True)
        files = self.controller.handle_select_data_folder(folder)
        self.window["-DATA FILE LIST-"].update(files)
        
    def handle_select_2d_data_folder_event(self, event, values):
        folder = values["-DATA 2D FOLDER-"]
        files = self.controller.handle_select_data_folder(folder)
        self.window["-DATA 2D FILE LIST-"].update(files)
        
    def handle_select_classifier_folder_event(self, event, values):
        folder = values["-CLASSIFIER FOLDER-"]
        f = self.controller.handle_select_classifier_folder(folder)
        self.window["-CLASSIFIER FILE LIST-"].update(f)
    
    def handle_dbm_technique_event(self, event, values):
        dbm_technique = values["-DBM TECHNIQUE-"]
        self.logger.log(f"DBM technique: {dbm_technique}")
        
        if dbm_technique == DBM_NNINV_TECHNIQUE:
            self.switch_visibility(["-PROJECTION TECHNIQUE COLUMN-"], True)
            return
        
        self.switch_visibility(["-PROJECTION TECHNIQUE COLUMN-", "-DATA 2D COLUMN-"], False)
        
    def handle_projection_technique_event(self, event, values):
        projection_technique = values["-PROJECTION TECHNIQUE-"]
        self.logger.log(f"Projection technique: {projection_technique}")
       
        if projection_technique == CUSTOM_PROJECTION_TECHNIQUE:
            self.switch_visibility(["-DATA 2D COLUMN-"], True)
            return
        
        self.switch_visibility(["-DATA 2D COLUMN-"], False)   
        
    def handle_changes_in_dbm_plotter(self):
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
        try:
            self.dbm_plotter_gui = self.controller.handle_get_decision_boundary_mapping_event(event, values)
            self.handle_changes_in_dbm_plotter()
        except Exception as e:
            self.gui_logger.error(e)
            # update loading state
            self.switch_visibility(["-DBM TEXT-", "-DBM IMAGE LOADING-","-DBM IMAGE-"], False)

        
    def handle_dbm_image_event(self, event, values):
        self.logger.log("Clicked on the dbm image")
        if self.dbm_plotter_gui is None:
            self.logger.warn("Nothing to show")
            return
        self.dbm_plotter_gui.start()

    def set_data_shape(self, data_shape: tuple):
        """This method sets the data shape of the data to be used for the DBM
        If the user imports data from a file (e.g. csv file), the data shape is unknown.
        This method is used to set the data shape after the data is imported, so that the data can be reshaped into the correct shape.
        Args:
            data_shape (tuple): The shape of the data
        """
        self.controller.data_shape = data_shape