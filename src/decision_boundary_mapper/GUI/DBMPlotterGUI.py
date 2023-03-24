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

import json
from math import sqrt
from datetime import datetime
from matplotlib.patches import Patch, Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import PySimpleGUI as sg
import os

from .. import Logger, LoggerGUI

def draw_figure_to_canvas(canvas, figure, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    toolbar = NavigationToolbar2Tk(figure_canvas_agg, canvas_toolbar)        
    toolbar.update()  
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def generate_color_mapper():
    colors_mapper = {
        -2: [0,0,0], # setting original test data to black
        -1: [1,1,1], # setting original train data to white
        0: [1,0,0], 
        1: [0,1,0], 
        2: [0,0,1], 
        3: [1,1,0], 
        4: [0,1,1], 
        5: [1,0,1], 
        6: [0.5,0.5,0.5], 
        7: [0.5,0,0], 
        8: [0,0.5,0], 
        9: [0,0,0.5]
    }
    # setting the rest of the colors 
    for i in range(10, 100):
        colors_mapper[i] = [i/200,i/200,i/200]
    return colors_mapper

# Generating initial settings
COLORS_MAPPER = generate_color_mapper()
APP_FONT = 'Helvetica 12'
TITLE = "Decision Boundary Map & Errors"
WINDOW_SIZE = (1550, 950)
BLACK_COLOR = "#252526"
BUTTON_PRIMARY_COLOR = "#007acc"
WHITE_COLOR = "#ffffff"
RIGHTS_MESSAGE_1 = "© 2023 Cristian Grosu. All rights reserved."
RIGHTS_MESSAGE_2 = "Made by Cristian Grosu for Utrecht University Master Thesis in 2023"
INFORMATION_CONTROLS_MESSAGE = "To change label(s) of a data point click on the data point, or select the data point by including them into a circle.\nPress any digit key to indicate the new label.\nPress 'Enter' to confirm the new label. Press 'Esc' to cancel the action.\nTo remove a change just click on the data point.\nAfter the changes are done press 'Apply Changes' to update the model. \nAfter the changes are applied the window will update."
DBM_WINDOW_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "dbm_plotter_icon.png")
CLASSIFIER_PERFORMANCE_HISTORY_FILE = "classifier_performance.log"
LABELS_CHANGES_FILE = "label_changes.json"

class DBMPlotterGUI:
    def __init__ (self, 
                  dbm_model,
                  img, img_confidence,
                  X_train, Y_train, 
                  X_test, Y_test,
                  encoded_train, encoded_test,                                    
                  save_folder,
                  projection_technique=None,
                  logger=None,
                  main_gui=None,
                  ):        
        """[summary] DBMPlotterGUI is a GUI that allows the user to visualize the decision boundary map and the errors of the DBM model.
        It also allows the user to change the labels of the data points and see the impact of the changes on the model.

        Args:
            dbm_model (DBM or SDBM): The DBM model that will be used to generate the decision boundary map.
            img (np.ndarray): The decision boundary map image.
            img_confidence (np.ndarray): The decision boundary map confidence image.
            X_train (np.ndarray): The training data.
            Y_train (np.ndarray): The training labels.
            X_test (np.ndarray): The test data.
            Y_test (np.ndarray): The test labels.
            encoded_train (np.ndarray): Positions of the training data points in the 2D embedding space.
            encoded_test (np.ndarray): Positions of the test data points in the 2D embedding space.
            spaceNd (np.ndarray): This is a list of lists of size resolution*resolution. Each point in this list is an nd data point which can be a vector, a matrix or a multidimensional matrix.
            save_folder (string): The folder where all the DBM model related files will be saved.
            projection_technique (string, optional): The projection technique the user wants to use if DBM is used as dbm_model. Defaults to None.
            logger (Logger, optional): The logger which is meant for logging the info messages. Defaults to None.
            main_gui (GUI, optional): The GUI that started the DBMPlotterGUI if any. Defaults to None.
        """
        if logger is None:
            self.console = Logger(name="DBMPlotterGUI")
        else:
            self.console = logger
        
        self.main_gui = main_gui # reference to main window  
        
        self.positions_of_labels_changes = ([], [], [])
        self.inverse_projection_errors = None
        self.projection_errors = None     
        
        self.initialize(dbm_model, 
                        img, 
                        img_confidence, 
                        X_train, Y_train, 
                        X_test, Y_test, 
                        encoded_train, encoded_test, 
                        save_folder, 
                        projection_technique)
    
    def initialize(self,
                   dbm_model,
                   img, img_confidence,
                   X_train, Y_train, 
                   X_test, Y_test,
                   encoded_train, encoded_test,                   
                   save_folder,
                   projection_technique=None,                   
                   ):
        self.dbm_model = dbm_model 
        self.img = img
        self.img_confidence = img_confidence
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.encoded_train = encoded_train
        self.encoded_test = encoded_test  
        self.save_folder = save_folder # folder where to save the changes made to the data by the user    
        self.projection_technique = projection_technique # projection technique used to generate the DBM                                    
        self.color_img, self.legend = self._build_2D_image_(img)
        self.train_mapper, self.test_mapper = self._generate_encoded_mapping_()
        
        # --------------------- Plotter related ---------------------                
        self.fig, self.ax = self._build_plot_()        
        self._build_annotation_mapper_()
        
        self.fig.legend(handles=self.legend, borderaxespad=0. )
        # --------------------- Plotter related ---------------------
        
        # --------------------- Others ------------------------------           
        self.motion_event_cid = None
        self.click_event_cid = None
        self.key_event_cid = None
        self.release_event_cid = None
        self.current_selected_point = None
        self.current_selected_point_assigned_label = None
        self.expert_updates_labels_mapper = {}        
        self.stop_application = False 
        self.update_labels_by_circle_select = True 
        self.update_labels_circle = None  
        
    def _initialize_gui_(self):        
        # --------------------- GUI related ---------------------
        self.window = self._build_GUI_()
        self.canvas, self.fig_agg, self.canvas_controls = self._build_canvas_(self.fig, key = "-DBM CANVAS-", controls_key="-CONTROLS CANVAS-")
        
        # --------------------- Classifier related ---------------------
        self.compute_classifier_metrics()
        
        self.draw_dbm_img()            
        # --------------------- GUI related ---------------------        
        self.updates_logger = LoggerGUI(name = "Updates logger", output = self.window["-LOGGER-"], update_callback = self.window.refresh)
        self.dbm_model.console = self.updates_logger
   
    def _get_GUI_layout_(self):   
        buttons_proj_errs = []
        buttons_inv_proj_errs = []        
        computed_projection_errors = self.projection_errors is not None
        computed_inverse_projection_errors = self.inverse_projection_errors is not None
        if not computed_projection_errors: 
            buttons_proj_errs = [
                sg.Button('Compute Projection Errors (interpolation)', font=APP_FONT, expand_x=True, key="-COMPUTE PROJECTION ERRORS INTERPOLATION-", button_color=(WHITE_COLOR, BUTTON_PRIMARY_COLOR)),
                sg.Button('Compute Projection Errors (inverse projection)', font=APP_FONT, expand_x=True, key="-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-", button_color=(WHITE_COLOR, BUTTON_PRIMARY_COLOR))             
            ]
        if not computed_inverse_projection_errors:
            buttons_inv_proj_errs = [
                sg.Button('Compute Inverse Projection Errors', font=APP_FONT, expand_x=True, key="-COMPUTE INVERSE PROJECTION ERRORS-", button_color=(WHITE_COLOR, BUTTON_PRIMARY_COLOR)) 
            ]        
         
        layout = [                                  
                    [sg.Canvas(key='-DBM CANVAS-', expand_x=True, expand_y=True, pad=(0,0))],     
                    [sg.Canvas(key='-CONTROLS CANVAS-', expand_x=True, pad=(0,0))],
                    [
                        sg.Column([
                            [
                                sg.Button("Show classifier performance history", font=APP_FONT, expand_x=True, key="-SHOW CLASSIFIER PERFORMANCE HISTORY-", button_color=(WHITE_COLOR, BUTTON_PRIMARY_COLOR)),
                                sg.Text("Classifier accuracy: ", font=APP_FONT, expand_x=True, key="-CLASSIFIER ACCURACY-"),  
                            ],
                            [  
                                sg.Checkbox("Change labels by selecting with circle", default=True, key="-CIRCLE SELECTING LABELS-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0,0)),
                                sg.Checkbox("Show labels changes", default=False, key="-SHOW LABELS CHANGES-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0,0)),
                            ],
                            [
                                sg.Checkbox("Show dbm color map", default=True, key="-SHOW DBM COLOR MAP-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0,0)),
                                sg.Checkbox("Show dbm confidence", default=True, key="-SHOW DBM CONFIDENCE-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0,0)),                                
                            ],
                            [
                                sg.Checkbox("Show inverse projection errors", default=False, key="-SHOW INVERSE PROJECTION ERRORS-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0,0), visible=computed_inverse_projection_errors),
                                sg.Checkbox("Show projection errors", default=False, key="-SHOW PROJECTION ERRORS-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0,0), visible=computed_projection_errors),
                            ],
                            #[
                            #    sg.Text("Epochs to train: ", font=APP_FONT, expand_x=True, pad=(0,0), key="-EPOCHS LABEL-"),
                            #    sg.Input("2", font=APP_FONT, key="-EPOCHS-", background_color=WHITE_COLOR, text_color=BLACK_COLOR, expand_x=True, justification="center"),                            
                            #],
                            [
                                sg.Button('Apply Updates', font=APP_FONT, expand_x=True, key="-APPLY CHANGES-", button_color=(WHITE_COLOR, BUTTON_PRIMARY_COLOR)),
                            ],
                            buttons_proj_errs,
                            buttons_inv_proj_errs,
                            [sg.Text(INFORMATION_CONTROLS_MESSAGE, expand_x=True)],
                            [sg.Text(RIGHTS_MESSAGE_1, expand_x=True)],
                            [sg.Text(RIGHTS_MESSAGE_2, expand_x=True)],           
                        ], expand_x=True),
                        sg.Column([
                            [sg.Multiline("", key="-LOGGER-", size=(50,20), background_color=WHITE_COLOR, text_color=BLACK_COLOR, auto_size_text=True, expand_y=True, expand_x=True)],                    
                        ], expand_x=True),
                    ]                                                      
                ]
        
        return layout
        
    def _build_GUI_(self):        
        window = sg.Window(title = TITLE, 
                           layout = self._get_GUI_layout_(), 
                           size=WINDOW_SIZE, 
                           resizable=True,
                           icon=DBM_WINDOW_ICON_PATH,
                           element_justification='center',
                           )
        window.finalize()
        #window.maximize()
        return window
    
    def start(self):
        self._initialize_gui_()
        while True:
            event, values = self.window.read()
            
            if event == "Exit" or event == sg.WIN_CLOSED or self.stop_application:
                break
        
            self.handle_event(event, values)            
        
        self.stop()
    
    def stop(self):
        if self.main_gui is not None and hasattr(self.main_gui, "handle_changes_in_dbm_plotter"):
            self.main_gui.handle_changes_in_dbm_plotter()
        self.console.log("Closing the application...")
        
        classifier_performance_path = os.path.join(self.save_folder, CLASSIFIER_PERFORMANCE_HISTORY_FILE)
        labels_changes_path = os.path.join(self.save_folder, LABELS_CHANGES_FILE)
        
        if os.path.exists(classifier_performance_path):
            os.remove(classifier_performance_path)
        
        if os.path.exists(labels_changes_path):
            os.remove(labels_changes_path)
                    
        self.window.close()
    
    def handle_event(self, event, values):
        EVENTS = {
            "-COMPUTE PROJECTION ERRORS INTERPOLATION-": self.handle_compute_projection_errors_event,
            "-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-": self.handle_compute_projection_errors_event,
            "-COMPUTE INVERSE PROJECTION ERRORS-": self.handle_compute_inverse_projection_errors_event,
            "-APPLY CHANGES-": self.handle_apply_changes_event,
            "-SHOW DBM COLOR MAP-": self.handle_checkbox_change_event,
            "-SHOW DBM CONFIDENCE-": self.handle_checkbox_change_event,
            "-SHOW INVERSE PROJECTION ERRORS-": self.handle_checkbox_change_event,
            "-SHOW PROJECTION ERRORS-": self.handle_checkbox_change_event,
            "-SHOW LABELS CHANGES-": self.handle_checkbox_change_event,
            "-CIRCLE SELECTING LABELS-": self.handle_circle_selecting_labels_change_event,            
            "-SHOW CLASSIFIER PERFORMANCE HISTORY-": self.handle_show_classifier_performance_history_event,
        }
        
        EVENTS[event](event, values)    
    
    def _build_2D_image_(self, img, class_name_mapper = lambda x: str(x), colors_mapper = COLORS_MAPPER):
        """Combines the img and the img_confidence into a single image.

        Args:
            img (np.ndarray): Label image.
            img_confidence (np.ndarray): Confidence image.
            class_name_mapper (function, optional): Describes how to map the data labels. Defaults to lambdax:str(x).
            colors_mapper (dict, optional): Mapper of labels to colors. Defaults to COLORS_MAPPER.

        Returns:
            np.ndarray: The combined image.
            patches: The legend patches.
        """
        color_img = np.zeros((img.shape[0], img.shape[1], 3))
                
        for i, j in np.ndindex(img.shape):
            color_img[i][j] = colors_mapper[img[i][j]]
        values = np.unique(img)
        
        patches = []
        for value in values:
            color = colors_mapper[value]
            if value==-1:
                label = "Original train data"
            elif value==-2:
                label = "Original test data"
            else:
                label = f"Value region: {class_name_mapper(value)}"

            patches.append(Patch(color=color, label=label))
        
        return color_img, patches
   
    def _generate_encoded_mapping_(self):
        """Generates a mapping of the encoded data to the original data.

        Returns:
            train_mapper (dict): Mapping of the encoded train data to the original train data.
            test_mapper (dict): Mapping of the encoded test data to the original test data.
        """
        train_mapper = {}
        for k in range(len(self.encoded_train)):
            [x, y] = self.encoded_train[k]
            train_mapper[f"{x} {y}"] = k
    
        test_mapper = {}
        for k in range(len(self.encoded_test)):
            [x, y] = self.encoded_test[k]
            test_mapper[f"{x} {y}"] = k
        
        return train_mapper, test_mapper

    def _build_plot_(self):                   
        fig = figure.Figure(figsize = (1, 1))
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        return fig, ax
    
    def _build_canvas_(self, fig, key, controls_key):        
        canvas = self.window[key].TKCanvas
        canvas_controls = self.window[controls_key].TKCanvas
        fig_agg = draw_figure_to_canvas(canvas, fig, canvas_controls)
        return canvas, fig_agg, canvas_controls
    
    def _build_annotation_mapper_(self):
        """ Builds the annotation mapper.
            This is used to display the data point label when hovering over the decision boundary.
        """
        image = OffsetImage(self.X_train[0], zoom=2, cmap="gray")
        label = TextArea("Data point label: None")
        
        annImage = AnnotationBbox(image, (0,0), xybox=(50., 50.), xycoords='data', boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))        
        annLabels = AnnotationBbox(label, (0,0), xybox=(50., 50.), xycoords='data', boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

        self.ax.add_artist(annImage)
        self.ax.add_artist(annLabels)
        annImage.set_visible(False)
        annLabels.set_visible(False)

        fig_width, fig_height = self.fig.get_size_inches() * self.fig.dpi
          
        def display_annotation(event):
            """Displays the annotation box when hovering over the decision boundary based on the mouse position."""
            if event.inaxes == None:
                return
        
            j, i = int(event.xdata), int(event.ydata)
            
            # change the annotation box position relative to mouse.
            ws = (event.x > fig_width/2.)*-1 + (event.x <= fig_width/2.) 
            hs = (event.y > fig_height/2.)*-1 + (event.y <= fig_height/2.)
            annImage.xybox = (50. * ws, 50. * hs)
            annLabels.xybox = (-50. * ws, 50. *hs)
            # make annotation box visible
            annImage.set_visible(True)
            # place it at the position of the event scatter point
            annImage.xy = (j, i)
            
            x_data, y_data = find_data_point(i,j)                            
            if x_data is not None:
                annImage.set_visible(True)
                image.set_data(x_data)
            else:
                annImage.set_visible(False)
            
            if y_data is not None:    
                annLabels.set_visible(True)
                annLabels.xy = (j, i)
                label.set_text(f"{y_data}")
            else:
                annLabels.set_visible(False)

            self.fig.canvas.draw_idle()
            
        def find_data_point(i, j):
            # search for the data point in the encoded train data
            if self.img[i][j] == -1:
                k = self.train_mapper[f"{i} {j}"]
                if f"{i} {j}" in self.expert_updates_labels_mapper:
                    l = self.expert_updates_labels_mapper[f"{i} {j}"][0]
                    return self.X_train[k], f"Label {self.Y_train[k]} \nExpert label: {l}"  
                return self.X_train[k], f"Label: {self.Y_train[k]}"
            
            # search for the data point in the encoded test data
            if self.img[i][j] == -2:
                k = self.test_mapper[f"{i} {j}"]
                return self.X_test[k], None      
            
            # search for the data point in the 
            point = None #self.spaceNd[i][j]
            return point, None
                    
        def onclick(event):
            """ Open the data point in a new window when clicked on a pixel in training or testing set based on mouse position."""            
            if event.inaxes == None:
                return
            
            if self.update_labels_by_circle_select:
                onclick_circle_strategy(event)
                return
            
            self.console.log("Clicked on: " + str(event.xdata) + ", " + str(event.ydata))
            j, i = int(event.xdata), int(event.ydata)

            if self.img[i][j] != -1:
                self.console.log("Data point not in training set")
                return
                    
            # check if clicked on a data point that already was updated, then remove the update
            if f"{i} {j}" in self.expert_updates_labels_mapper:
                (_, point) = self.expert_updates_labels_mapper[f"{i} {j}"]
                point.remove()
                del self.expert_updates_labels_mapper[f"{i} {j}"]                    
                self.fig.canvas.draw_idle()
                self.updates_logger.log(f"Removed updates for point: ({j}, {i})")
                return
            
            # if clicked on a data point that was not updated, then add the update
            self.current_selected_point = self.ax.plot(j, i, 'bo', markersize=5)[0]
            
            # disable annotations on hover
            #if self.motion_event_cid is not None:
            #    self.fig.canvas.mpl_disconnect(self.motion_event_cid)
            # disable on click event
            if self.click_event_cid is not None:
                self.fig.canvas.mpl_disconnect(self.click_event_cid)
                self.click_event_cid = None
            
            # enable key press events
            self.key_event_cid = self.fig.canvas.mpl_connect('key_press_event', onkey)
       
            self.fig.canvas.draw_idle()
        
        def onkey(event):                                           
            if self.current_selected_point is None:
                return
            
            if event.key.isdigit():
                self.current_selected_point_assigned_label = int(event.key)
                return
            
            (x, y) = self.current_selected_point.get_data()
            
            if event.key == 'escape' or event.key == 'enter':                            
                self.current_selected_point.remove()       
                
                if event.key == 'escape':                    
                    self.updates_logger.log("Cancelled point move...")                    
                    
                elif event.key == 'enter':
                    # showing the fix of the position                    
                    self.current_selected_point = self.ax.plot(x, y, 'b^')[0]                 
                    if self.current_selected_point_assigned_label is not None:
                        self.updates_logger.log(f"Assigned label: {self.current_selected_point_assigned_label} to point: ({x[0]}, {y[0]})")                        
                        self.expert_updates_labels_mapper[f"{y[0]} {x[0]}"] = (self.current_selected_point_assigned_label, self.current_selected_point)                        
                    
                self.current_selected_point = None
                self.current_selected_point_assigned_label = None
               
                self.motion_event_cid = self.fig.canvas.mpl_connect('motion_notify_event', display_annotation)                
                self.click_event_cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
                self.fig.canvas.mpl_disconnect(self.key_event_cid)  
                self.fig.canvas.draw_idle()  
                return
        
        def onclick_circle_strategy(event):                        
            self.console.log("Clicked on: " + str(event.xdata) + ", " + str(event.ydata))
            x, y = int(event.xdata), int(event.ydata)
            self.current_selected_point = (x,y)
            self.release_event_cid = self.fig.canvas.mpl_connect('button_release_event', onrelease_circle_strategy)
        
        def onrelease_circle_strategy(event):
            if event.inaxes == None:
                return
            self.console.log("Released on: " + str(event.xdata) + ", " + str(event.ydata))
            (x0, y0) = self.current_selected_point
            x1, y1 = int(event.xdata), int(event.ydata)
            (cx, cy) = (x0 + (x1 - x0)/2, y0 + (y1 - y0)/2) # circle center
            r = sqrt(((x1 - x0)/2)**2 + ((y1 - y0)/2)**2)   # circle radius
            self.update_labels_circle = Circle((cx, cy), r, color='black', fill=False)
            self.ax.add_artist(self.update_labels_circle)    
            self.key_event_cid = self.fig.canvas.mpl_connect('key_press_event', onkey_circle_strategy)    
            if self.release_event_cid is not None:
                self.fig.canvas.mpl_disconnect(self.release_event_cid)
            self.fig.canvas.draw_idle()

        def onkey_circle_strategy(event):
            if self.update_labels_circle is None:
                return

            if event.key.isdigit():
                self.current_selected_point_assigned_label = int(event.key)
                return
            
            (x, y) = self.update_labels_circle.get_center()
            r = self.update_labels_circle.get_radius()
            if event.key == "enter" and self.current_selected_point_assigned_label is not None:                                    
                self.updates_logger.log(f"Assigned label: {self.current_selected_point_assigned_label} to circle: center ({x}, {y}), radius ({r})")
                positions = find_points_in_circle((x,y), r)
                for pos in positions:
                    point = self.ax.plot(pos[0], pos[1], 'b^')[0]   
                    self.expert_updates_labels_mapper[f"{pos[1]} {pos[0]}"] = (self.current_selected_point_assigned_label, point)                        
                           
            if event.key == "backspace":
                self.updates_logger.log(f"Removing changes in circle: center ({x},{y}), radius ({r})")
                positions = find_points_in_circle((x,y), r)
                for pos in positions:
                    if f"{pos[1]} {pos[0]}" in self.expert_updates_labels_mapper:
                        self.expert_updates_labels_mapper[f"{pos[1]} {pos[0]}"][1].remove()
                        del self.expert_updates_labels_mapper[f"{pos[1]} {pos[0]}"]
                
            self.update_labels_circle.remove()
            self.update_labels_circle = None
            self.fig.canvas.mpl_disconnect(self.key_event_cid)  
            self.fig.canvas.draw_idle()
        
        def find_points_in_circle(circle_center, circle_radius):
            (cx, cy) = circle_center
            initial_x, initial_y = int(cx - circle_radius), int(cy - circle_radius)
            final_x, final_y = int(cx + circle_radius) + 1, int(cy + circle_radius) + 1
            initial_x = 0 if initial_x < 0 else initial_x
            initial_y = 0 if initial_y < 0 else initial_y
            final_x = self.img.shape[0] if final_x > self.img.shape[0] else final_x
            final_y = self.img.shape[0] if final_y > self.img.shape[0] else final_y
            
            positions = []
            for x in range(initial_x, final_x):
                for y in range(initial_y, final_y):
                    if (self.img[y, x] == -1) and ((x - cx)**2 + (y - cy)**2 <= circle_radius**2):
                        positions.append((x,y))
            return positions
            
                    
        self.motion_event_cid = self.fig.canvas.mpl_connect('motion_notify_event', display_annotation)           
        self.click_event_cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
       
    def plot_data_point(self, data, label):
        """Plots the data point in a new window.

        Args:
            data (np.ndarray): the data point to be displayed
            label (str): the label of the data point
        """
        img = Image.fromarray(np.uint8(data * 255))
        img.thumbnail((100, 100), Image.ANTIALIAS)
        img.show(title=f"Data point label: {label}")
    
    def draw_dbm_img(self):             
        # update the figure
        self.ax.set_title("Decision Boundary Mapper")    
        img = np.zeros((self.img.shape[0], self.img.shape[1], 4))
        img[:,:,:3] = self.color_img
        img[:,:,3] = self.img_confidence    
        self.axes_image = self.ax.imshow(img)                                    
                    
        # draw the figure to the canvas
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)    
        self.window.refresh()
        
    def compute_classifier_metrics(self):
        self.console.log("Evaluating classifier...")
        loss, accuracy = self.dbm_model.classifier.evaluate(self.X_test, self.Y_test, verbose=0)        
        self.console.log(f"Classifier Accuracy: {(100 * accuracy):.2f}%  Loss: {loss:.2f}")
        
        path = os.path.join(self.save_folder, CLASSIFIER_PERFORMANCE_HISTORY_FILE)
        
        with open(path, "a") as f:
            time = datetime.now().strftime("%D %H:%M:%S")
            message = f"Accuracy: {(100 * accuracy):.2f}%  Loss: {loss:.2f}"
            f.write(f"{time} {message}\n")
            
        self.window["-CLASSIFIER ACCURACY-"].update(f"Classifier Accuracy: {(100 * accuracy):.2f} %  Loss: {loss:.2f}")
    
    def handle_compute_inverse_projection_errors_event(self, event, values):
        self.window['-COMPUTE INVERSE PROJECTION ERRORS-'].update(visible=False, disabled=True)        
        self.updates_logger.log("Computing inverse projection errors, please wait...")
        possible_path = os.path.join(self.save_folder, "inverse_projection_errors.npy")
        # try to get projection errors from cache first
        if os.path.exists(possible_path):
            self.inverse_projection_errors = np.load(possible_path)
        else:
            self.inverse_projection_errors = self.dbm_model.generate_inverse_projection_errors(save_folder=self.save_folder, resolution = len(self.img))

        self.updates_logger.log("Inverse projection errors computed!")
        self.window['-SHOW INVERSE PROJECTION ERRORS-'].update(visible=True)
            
    def _set_loading_proj_errs_state_(self):
        self.window['-COMPUTE PROJECTION ERRORS INTERPOLATION-'].update(visible=False, disabled=True)        
        self.window['-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-'].update(visible=False, disabled=True)        
        self.updates_logger.log("Computing projection errors, please wait...")
        
    def handle_compute_projection_errors_event(self, event, values):
        self._set_loading_proj_errs_state_()
        if event == "-COMPUTE PROJECTION ERRORS INTERPOLATION-":
            possible_path = os.path.join(self.save_folder, "projection_errors_interpolated.npy")
            # try to get projection errors from cache first
            if os.path.exists(possible_path):
                self.projection_errors = np.load(possible_path)
            else:
                self.projection_errors = self.dbm_model.generate_projection_errors(use_interpolation=True, save_folder=self.save_folder)
        elif event == "-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-":            
            possible_path = os.path.join(self.save_folder, "projection_errors_inv_proj.npy")
            # try to get projection errors from cache first
            if os.path.exists(possible_path):
                self.projection_errors = np.load(possible_path)
            else:
                self.projection_errors = self.dbm_model.generate_projection_errors(use_interpolation=False, save_folder=self.save_folder)        
        
        self.updates_logger.log("Finished computing projection errors.")
        self.window['-SHOW PROJECTION ERRORS-'].update(visible=True)           

    def handle_checkbox_change_event(self, event, values):
        show_color_map = values["-SHOW DBM COLOR MAP-"]
        show_confidence = values["-SHOW DBM CONFIDENCE-"]
        show_inverse_projection_errors = values["-SHOW INVERSE PROJECTION ERRORS-"]
        show_projection_errors = values["-SHOW PROJECTION ERRORS-"]
        show_labels_changes = values["-SHOW LABELS CHANGES-"]
        
        color_img = np.zeros((self.img.shape[0], self.img.shape[1], 3))
        alphas = 1 + np.zeros((self.img.shape[0], self.img.shape[1]))
        img = np.zeros((self.img.shape[0], self.img.shape[1], 4))
        
        if show_color_map:
            color_img = self.color_img
        
        if show_confidence:
            alphas = alphas * self.img_confidence
        if show_inverse_projection_errors:
            alphas = alphas * (1 - self.inverse_projection_errors)
            
        if show_projection_errors and self.projection_errors is not None:
            alphas = alphas * (1 - self.projection_errors)
        
        img[:, :, :3] = color_img
        img[:, :, 3] = alphas                     
        
        if hasattr(self, "axes_image"):
            self.axes_image.remove()
        self.axes_image = self.ax.imshow(img) 
        
        if hasattr(self, "ax_labels_changes") and self.ax_labels_changes is not None:            
            self.ax_labels_changes.set_visible(False)
            #for point in self.ax_labels_changes:                
            #    point.remove()
            self.ax_labels_changes = None
        
        positions_x, positions_y, alphas = self.positions_of_labels_changes   
        if show_labels_changes and len(positions_x) > 0 and len(positions_y) > 0  and len(alphas) > 0:                             
            self.ax_labels_changes = self.ax.scatter(positions_x, positions_y, c='green', marker='^', alpha=alphas)   
        
        self.fig.canvas.draw_idle()
    
    def handle_circle_selecting_labels_change_event(self, event, values):
        self.update_labels_by_circle_select = values["-CIRCLE SELECTING LABELS-"]
        
    def handle_apply_changes_event(self, event, values):
        num_changes = len(self.expert_updates_labels_mapper)
        if num_changes == 0:
            self.updates_logger.log("No changes to apply")
            return
        
        if num_changes < 10:
            self.updates_logger.log("Less than 10 changes to apply, please apply more changes")
            return
        
        """
        if values["-EPOCHS-"].isdigit():
            epochs = int(values["-EPOCHS-"])
        else:
            epochs = 2
            self.window["-EPOCHS-"].update(epochs)
        """
        epochs = 2
        
        self.console.log("Transforming changes...")
        Y, label_changes = self.transform_changes()
        
        self.console.log("Saving changes to a local folder...")
        self.save_changes(self.save_folder, label_changes=label_changes)
        
        self.updates_logger.log("Applying changes... This might take a couple of seconds, after this the window will be closed")
                
        self.dbm_model.refit_classifier(self.X_train, Y, save_folder=os.path.join(self.save_folder, "refit_classifier"), epochs = epochs)
        
        # Updating the main GUI with the new model                
        dbm_info = self.dbm_model.generate_boundary_map(
            self.X_train, 
            Y, 
            self.X_test, 
            self.Y_test, 
            resolution=len(self.img),
            use_fast_decoding=True,
            load_folder=self.save_folder,
            projection=self.projection_technique                                        
        )        
            
        img, img_confidence, encoded_training_data, encoded_testing_data, training_history = dbm_info
        self.initialize(dbm_model = self.dbm_model,
                            img = img,
                            img_confidence = img_confidence,
                            encoded_train = encoded_training_data, 
                            encoded_test = encoded_testing_data,
                            X_train = self.X_train, 
                            Y_train = Y,
                            X_test = self.X_test,
                            Y_test = self.Y_test,                            
                            save_folder=self.save_folder,
                            projection_technique=self.projection_technique,
                        )
        
        self.draw_dbm_img()   
        self.handle_checkbox_change_event(event, values)      
        self.compute_classifier_metrics()
        
        self.updates_logger.log("Changes applied successfully!")     
            
    def save_changes(self, folder:str="tmp", label_changes={}):
        if not os.path.exists(folder):
            os.path.makedirs(folder)                        
        
        with open(os.path.join(folder, LABELS_CHANGES_FILE), "a") as f:
            f.write("\n" + "-" * 50 + "\n")
            json.dump(label_changes, f, indent=2)
    
    def transform_changes(self):
        """
        Returns:
            position_changes (dict): a dictionary with old position as key and the new position as value
            label_changes (dict): a dictionary with old and new positions as key and the new label as value
        """     
        Y = np.copy(self.Y_train) 
        labels_changes = {}
        
        pos_x, pos_y, alphas = self.positions_of_labels_changes
        
        ALPHA_DECAY_ON_UPDATE = 0.05        
        alphas = [alpha - ALPHA_DECAY_ON_UPDATE if alpha > ALPHA_DECAY_ON_UPDATE else ALPHA_DECAY_ON_UPDATE for alpha in alphas]
        
        for pos in self.expert_updates_labels_mapper:
            k = self.train_mapper[pos]
            pos_x.append(int(pos.split(" ")[1]))
            pos_y.append(int(pos.split(" ")[0]))
            alphas.append(1)
            y = self.expert_updates_labels_mapper[pos][0]
            Y[k] = y
            labels_changes[pos] = self.expert_updates_labels_mapper[pos][0]
        
        self.positions_of_labels_changes = (pos_x, pos_y, alphas)
            
        return Y, labels_changes
    
    def handle_show_classifier_performance_history_event(self, event, values):
        path = os.path.join(self.save_folder, CLASSIFIER_PERFORMANCE_HISTORY_FILE)
        if not os.path.isfile(path):
            return
        
        times, accuracies, losses = [], [], []
        with open(path, "r") as f:
            for line in f.readlines():                
                line = line.strip()
                if len(line) == 0:
                    continue
                _, time, _, acc, _, _, loss = line.replace("\n", "").replace("%", "").split(" ")                        
                times.append(time)
                accuracies.append(float(acc))
                losses.append(float(loss))
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        
        ax1.set_title("Classifier accuracy history")
        ax2.set_title("Classifier loss history")
        ax1.set_xlabel("Time")
        ax2.set_xlabel("Time")
        ax1.set_ylabel("Accuracy (%)")
        ax2.set_ylabel("Loss")
        
        ax1.plot(times, accuracies)
        ax2.plot(times, losses)
        plt.show()
        