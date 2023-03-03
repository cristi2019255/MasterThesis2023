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
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import matplotlib.figure as figure
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
INFORMATION_CONTROLS_MESSAGE = "To change the position of a data point click on the data point, then use the arrow keys to move it. \nPress 'Enter' to confirm the new position. Press 'Esc' to cancel the movement.\nTo remove a change just click on the data point.\nAfter the changes are done press 'Apply Changes' to update the model. \nAfter the changes are applied the window will close.\nTo see the changes just reopen this window from the previous window."
DBM_WINDOW_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "dbm_plotter_icon.png")

class DBMPlotterGUI:
    def __init__ (self, 
                  dbm_model,
                  img, img_confidence,
                  X_train, Y_train, 
                  X_test, Y_test,
                  encoded_train, encoded_test, spaceNd,                   
                  save_folder,
                  projection_technique=None,
                  logger=None,
                  main_gui=None,
                  ):        
        
        if logger is None:
            self.console = Logger(name="DBMPlotterGUI")
        else:
            self.console = logger
        
        self.dbm_model = dbm_model 
        self.img = img
        self.img_confidence = img_confidence
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.encoded_train = encoded_train
        self.encoded_test = encoded_test  
        self.main_gui = main_gui # reference to main window  
        self.save_folder = save_folder # folder where to save the changes made to the data by the user    
        self.projection_technique = projection_technique # projection technique used to generate the DBM                            
        self.spaceNd = spaceNd.reshape(spaceNd.shape[0], spaceNd.shape[1], X_train.shape[1], X_train.shape[2])
        self.color_img, self.legend = self._build_2D_image_(img, img_confidence)
        self.train_mapper, self.test_mapper = self._generate_encoded_mapping_()
        self.inverse_projection_errors = self.dbm_model.generate_inverse_projection_errors()
        
        # --------------------- Plotter related ---------------------                
        self.fig, self.ax = self._build_plot_()        
        self._build_annotation_mapper_()
        
        self.fig.legend(handles=self.legend, borderaxespad=0. )
        # --------------------- Plotter related ---------------------
        
        # --------------------- Others ------------------------------
        self.proj_errs_computed = False
        self.motion_event_cid = None
        self.click_event_cid = None
        self.key_event_cid = None
        self.current_selected_point = None
        self.current_selected_point_assigned_label = None
        self.expert_updates_labels_mapper = {}        
        self.stop_application = False    
        
    def _initialize_gui_(self):        
        # --------------------- GUI related ---------------------
        self.window = self._build_GUI_()
        self.canvas, self.fig_agg, self.canvas_controls = self._build_canvas_(self.fig, key = "-DBM CANVAS-", controls_key="-CONTROLS CANVAS-")
        
        self.draw_dbm_img()    
        # --------------------- GUI related ---------------------        
        self.updates_logger = LoggerGUI(name = "Updates logger", output = self.window["-LOGGER-"], update_callback = self.window.refresh)
        self.dbm_model.console = self.updates_logger
   
    def _get_GUI_layout_(self):   
        buttons = []
           
        if not self.proj_errs_computed: 
            buttons.append(sg.Button('Compute Projection Errors', font=APP_FONT, expand_x=True, key="-COMPUTE PROJECTION ERRORS-", button_color=(WHITE_COLOR, BUTTON_PRIMARY_COLOR)))             

        layout = [                                  
                    [sg.Canvas(key='-DBM CANVAS-', expand_x=True, expand_y=True, pad=(0,0))],     
                    [sg.Canvas(key='-CONTROLS CANVAS-', expand_x=True, pad=(0,0))],
                    [
                        sg.Column([
                            [sg.Button('Apply Updates', font=APP_FONT, expand_x=True, key="-APPLY CHANGES-", button_color=(WHITE_COLOR, BUTTON_PRIMARY_COLOR))],
                            buttons,
                            [sg.Text(INFORMATION_CONTROLS_MESSAGE, expand_x=True)],
                            [sg.Text(RIGHTS_MESSAGE_1, expand_x=True)],
                            [sg.Text(RIGHTS_MESSAGE_2, expand_x=True)],           
                        ], expand_x=True),
                        sg.Column([
                            [sg.Multiline("",expand_x=True, size=(50,10), key="-LOGGER-", background_color=WHITE_COLOR, text_color=BLACK_COLOR, auto_size_text=True)],                    
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
        self.console.log("Closing the application...")
        self.window.close()
    
    def handle_event(self, event, values):
        EVENTS = {
            "-COMPUTE PROJECTION ERRORS-": self.handle_compute_projection_errors_event,
            "-APPLY CHANGES-": self.handle_apply_changes_event,
        }
        
        EVENTS[event](event, values)    
    
    def _build_2D_image_(self, img, img_confidence, class_name_mapper = lambda x: str(x), colors_mapper = COLORS_MAPPER):
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
        color_img = np.zeros((img.shape[0], img.shape[1], 4))
                
        for i, j in np.ndindex(img.shape):
            color_img[i,j] = colors_mapper[img[i,j]] + [img_confidence[i,j]]
        
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

            patches.append(mpatches.Patch(color=color, label=label))
        
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

        xybox=(50., 50.)
        annImage = AnnotationBbox(image, (0,0), xybox=xybox, xycoords='data',boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))        
        annLabels = AnnotationBbox(label, (0,0), xybox=xybox, xycoords='data', boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

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
            image.set_data(x_data)
            
            if y_data is not None:    
                annLabels.set_visible(True)
                annLabels.xy = (j, i)
                label.set_text(f"{y_data}")
            else:
                annLabels.set_visible(False)

            self.fig.canvas.draw_idle()
            
        def find_data_point(i, j):
            # search for the data point in the encoded train data
            if self.img[i,j] == -1:
                k = self.train_mapper[f"{i} {j}"]
                if f"{i} {j}" in self.expert_updates_labels_mapper:
                    l = self.expert_updates_labels_mapper[f"{i} {j}"][0]
                    return self.X_train[k], f"Label {self.Y_train[k]} \nExpert label: {l}"  
                return self.X_train[k], f"Label: {self.Y_train[k]}"
            
            # search for the data point in the encoded test data
            if self.img[i,j] == -2:
                k = self.test_mapper[f"{i} {j}"]
                if f"{i} {j}" in self.expert_updates_labels_mapper:
                    l = self.expert_updates_labels_mapper[f"{i} {j}"][0]
                    return self.X_test[k], f"Label {self.Y_test[k]} \nExpert label: {l}"  
                return self.X_test[k], f"Label: {self.Y_test[k]}"       
            
            # search for the data point in the 
            point = self.spaceNd[i,j]
            if f"{i} {j}" in self.expert_updates_labels_mapper:
                l = self.expert_updates_labels_mapper[f"{i} {j}"][0]
                return point, f"Expert label {l}"            
            return point, None
                    
        def onclick(event):
            """ Open the data point in a new window when clicked on a pixel in training or testing set based on mouse position."""
            if event.inaxes == None:
                return
            
            self.console.log("Clicked on: " + str(event.xdata) + ", " + str(event.ydata))
            j, i = int(event.xdata), int(event.ydata)

            if self.img[i,j] >= 0 or self.img[i,j] < -2:
                self.console.log("Data point not in training or testing set")
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
        self.ax.imshow(self.color_img)       
        # draw the figure to the canvas
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)    
        self.window.refresh()
    
    def _set_loading_proj_errs_state_(self):
        self.window['-COMPUTE PROJECTION ERRORS-'].update(visible=False, disabled=True)        
        self.updates_logger.log("Computing projection errors, please wait...")
        
    def handle_compute_projection_errors_event(self, event, values):
        self._set_loading_proj_errs_state_()
        projection_errors = self.dbm_model.generate_projection_errors()
        self.proj_errs_computed = True
        self.updates_logger.log("Finished computing projection errors.")

    #todo: change these function according to todos
    def handle_apply_changes_event(self, event, values):
        num_changes = len(self.expert_updates_labels_mapper)
        if num_changes == 0:
            self.updates_logger.log("No changes to apply")
            return
        # TODO: uncomment this
        #if num_changes < 10:
        #    self.updates_logger.log("Less than 10 changes to apply, please apply more changes")
        #    return
        
        self.console.log("Transforming changes...")
        Xnd, Y, label_changes = self.transform_changes()
        
        self.console.log("Saving changes to a local folder...")
        self.save_changes(self.save_folder, label_changes=label_changes)
        
        self.updates_logger.log("Applying changes... This might take a couple of seconds, after this the window will be closed")
        self.updates_logger.log("You can reopen the window from the main GUI (previous window)")
        
        # For the start lets do just the position changes (not the labels change)
        self.dbm_model.refit_classifier(Xnd, Y, save_folder=os.path.join(self.save_folder, "refit_classifier"))
        
        # TODO: update self.X_train, self.Y_train, self.X_test, self.Y_test with the new labels from Y
        
        # Updating the main GUI with the new model                
        dbm_info = self.dbm_model.generate_boundary_map(
            self.X_train, 
            self.Y_train, 
            self.X_test, 
            self.Y_test, 
            resolution=len(self.img),
            use_fast_decoding=False,
            load_folder=self.save_folder,
            projection=self.projection_technique                                        
        )        
        
        if self.main_gui is not None and hasattr(self.main_gui, "handle_changes_in_dbm_plotter"):
            self.main_gui.handle_changes_in_dbm_plotter(dbm_info, self.dbm_model, self.save_folder, self.projection_technique)
            self.stop_application=True
            return
        else:
            img, img_confidence, encoded_training_data, encoded_testing_data, spaceNd, training_history = dbm_info
            self.__init__(dbm_model = self.dbm_model,
                          img = img,
                          img_confidence = img_confidence,
                          encoded_train = encoded_training_data, 
                          encoded_test = encoded_testing_data,
                          X_train = self.X_train, 
                          Y_train = self.Y_train,
                          X_test = self.X_test,
                          Y_test = self.Y_test,
                          spaceNd=spaceNd,
                          save_folder=self.save_folder,
                          projection_technique=self.projection_technique,
                    )
            self.draw_dbm_img()
            return
            
    def save_changes(self, folder:str="tmp", label_changes={}):
        if not os.path.exists(folder):
            os.path.makedirs(folder)                        
        
        with open(os.path.join(folder, "label_changes.json"), "w") as f:
            json.dump(label_changes, f, indent=2)
    
    def transform_changes(self):
        """
        Returns:
            position_changes (dict): a dictionary with old position as key and the new position as value
            label_changes (dict): a dictionary with old and new positions as key and the new label as value
        """     
        Xnd = []
        Y = [] 
        labels_changes = {}
        
        for pos in self.expert_updates_labels_mapper:
            if pos in self.train_mapper:
                k = self.train_mapper[pos]
                xnd = self.X_train[k]
            else:
                k = self.test_mapper[pos]
                xnd = self.X_test[k]
            
            Xnd.append(xnd)
            Y.append(self.expert_updates_labels_mapper[pos][0])
            labels_changes[pos] = self.expert_updates_labels_mapper[pos][0]
            
        Xnd, Y = np.array(Xnd), np.array(Y)
        return Xnd, Y, labels_changes