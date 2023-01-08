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

TITLE = "Classifiers visualization tool"
WINDOW_SIZE = (1300, 800)
BACKGROUND_COLOR = "#252526"
BUTTON_PRIMARY_COLOR = "#007acc"
TEXT_COLOR = "#ffffff"
RIGHTS_MESSAGE = "© 2023 Cristian Grosu. All rights reserved."
RIGHTS_MESSAGE_2 = "Made by Cristian Grosu for Utrecht University Master Thesis in 2023"


import PySimpleGUI as sg
import os
from utils.Logger import Logger
from PIL import Image, ImageTk

class GUI:
    def __init__(self):
        self.window = self.build()
        self.logger = Logger(name = "GUI")
    
    def build(self):
        layout = self._get_layout()
        window = sg.Window(TITLE, layout, size=WINDOW_SIZE, background_color=BACKGROUND_COLOR)
        return window
        
    def start(self):
        while True:
            event, values = self.window.read()
            
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
        
            self.handle_event(event, values)
        
        self.stop()
    
    def stop(self):
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
                    [sg.Text("Decision boundary map: ", background_color=BACKGROUND_COLOR)],
                    [sg.Image(key="-DBM IMAGE-", size=(80, 80), visible=False, enable_events=True)],   
                ], background_color=BACKGROUND_COLOR),
            ],
            [
                sg.HSeparator(),
            ],
            [
                sg.Column([
                    [sg.Text("Logger: ", background_color=BACKGROUND_COLOR)],
                    [sg.Text("", size=(80, 20), key="-LOGGER-", background_color=TEXT_COLOR, text_color=BACKGROUND_COLOR, expand_y=True, auto_size_text=True)],   
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
    
    def handle_get_decision_boundary_mapping_event(self, event, values):
        default_dbm_image_path = os.path.join(os.getcwd(), "results", "MNIST", "2D_boundary_mapping.png")
        img = Image.open(default_dbm_image_path)
        img.thumbnail((600, 600), Image.ANTIALIAS)
        # Convert im to ImageTk.PhotoImage after window finalized
        image = ImageTk.PhotoImage(image=img)
        
        self.window["-DBM IMAGE-"].update(data=image)
        self.switch_visibility(["-DBM IMAGE-"], True)
    
    def handle_change_dbm_technique_event(self, event, values):
        self.logger.log("Changed dbm technique to: " + values["-DBM TECHNIQUE-"])
        
    def handle_dbm_image_event(self, event, values):
        self.logger.log("Clicked on the dbm image")