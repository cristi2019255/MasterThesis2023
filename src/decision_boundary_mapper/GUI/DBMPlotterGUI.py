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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from PIL import Image
import numpy as np
import PySimpleGUI as sg

from .. import Logger

def draw_figure_to_canvas(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
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
RIGHTS_MESSAGE = "© 2023 Cristian Grosu. All rights reserved."
RIGHTS_MESSAGE_2 = "Made by Cristian Grosu for Utrecht University Master Thesis in 2023"


class DBMPlotterGUI:
    def __init__ (self, 
                  dbm_model,
                  img, 
                  img_confidence,
                  X_train, 
                  Y_train, 
                  X_test, 
                  Y_test,
                  encoded_train, 
                  encoded_test,                                      
                  spaceNd, logger=None):
        
        
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
        self.spaceNd = spaceNd.reshape(spaceNd.shape[0], spaceNd.shape[1], X_train.shape[1], X_train.shape[2])
        self.color_img, self.legend = self._build_2D_image_(img, img_confidence)
        self.train_mapper, self.test_mapper = self._generate_encoded_mapping_()
        self.inverse_projection_errors = self.dbm_model.generate_inverse_projection_errors()
        
        # --------------------- Plotter related ---------------------                
        self.fig, self.dbm_ax, self.proj_errs_ax, self.inv_proj_errs_ax = self._build_plot_()        
        self._build_annotation_mapper_()
        
        ax_img = self.inv_proj_errs_ax.imshow(self.inverse_projection_errors)        
        self.fig.colorbar(ax_img, ax=self.inv_proj_errs_ax)
        self.fig.legend(handles=self.legend, borderaxespad=0. )
        # --------------------- Plotter related ---------------------
        self.proj_errs_computed = False    
        
        
    def _initialize_gui_(self):        
        # --------------------- GUI related ---------------------
        self.window = self._build_GUI_()
        self.canvas, self.fig_agg = self._build_canvas_(self.fig, key = "-DBM CANVAS-")
        
        self.draw_dbm_img()    
        # --------------------- GUI related ---------------------        
            
    def _get_GUI_layout_(self):   
        buttons = []
           
        if not self.proj_errs_computed: 
            buttons.append(sg.Button('Compute Projection Errors', font=APP_FONT, expand_x=True, key="-COMPUTE PROJECTION ERRORS-", button_color=(WHITE_COLOR, BUTTON_PRIMARY_COLOR)))             
      
        layout = [                                  
                    [
                        [sg.Canvas(key='-DBM CANVAS-', expand_x=True, expand_y=True, pad=(0,0))],                        
                    ], 
                    buttons,    
                    [sg.Text(RIGHTS_MESSAGE)],
                    [sg.Text(RIGHTS_MESSAGE_2)]              
                ]
        return layout
        
    def _build_GUI_(self):        
        window = sg.Window(title = TITLE, 
                           layout = self._get_GUI_layout_(), 
                           size=WINDOW_SIZE, 
                           resizable=True,
                           element_justification='center',
                           )
        window.finalize()
        return window
    
    def start(self):
        self._initialize_gui_()
        while True:
            event, values = self.window.read()
            
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
        
            self.handle_event(event, values)
        
        self.stop()
    
    def stop(self):
        self.console.log("Closing the application...")
        self.window.close()
    
    def handle_event(self, event, values):
        EVENTS = {
            "-COMPUTE PROJECTION ERRORS-": self.handle_compute_projection_errors_event,
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
        fig = plt.figure(figsize = (1, 1))
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(224)
        ax.set_axis_off()
        ax1.set_axis_off()
        ax2.set_axis_off()       
        return fig, ax, ax1, ax2
    
    def _build_canvas_(self, fig, key):        
        canvas = self.window[key].TKCanvas
        fig_agg = draw_figure_to_canvas(canvas, fig)
        return canvas, fig_agg
    
    def _build_annotation_mapper_(self):
        """ Builds the annotation mapper.
            This is used to display the data point label when hovering over the decision boundary.
        """
        image = OffsetImage(self.X_train[0], zoom=2, cmap="gray")
        label = TextArea("Data point label: None")

        xybox=(50., 50.)
        annImage = AnnotationBbox(image, (0,0), xybox=xybox, xycoords='data',boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))
        annImage_ax1 = AnnotationBbox(image, (0,0), xybox=xybox, xycoords='data', boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))
        annImage_ax2 = AnnotationBbox(image, (0,0), xybox=xybox, xycoords='data', boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))

        annLabels = AnnotationBbox(label, (0,0), xybox=xybox, xycoords='data', boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

        self.dbm_ax.add_artist(annImage)
        self.dbm_ax.add_artist(annLabels)
        self.inv_proj_errs_ax.add_artist(annImage_ax1)
        self.proj_errs_ax.add_artist(annImage_ax2)
        annImage.set_visible(False)
        annImage_ax1.set_visible(False)
        annImage_ax2.set_visible(False)
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
            annImage_ax1.set_visible(True)
            annImage_ax2.set_visible(True)
            # place it at the position of the event scatter point
            annImage.xy = (j, i)
            annImage_ax1.xy = (j, i)
            annImage_ax2.xy = (j, i)
            
            x_data, y_data = find_data_point(i,j)                            
            image.set_data(x_data)
            
            if y_data is not None:    
                annLabels.set_visible(True)
                annLabels.xy = (j, i)
                label.set_text(f"Label: {y_data}")
            else:
                annLabels.set_visible(False)

            self.fig.canvas.draw_idle()
                
        def onclick(event):
            """ Open the data point in a new window when clicked on a pixel in training or testing set based on mouse position."""
            if event.inaxes == None:
                return
            
            self.console.log("Clicked on: " + str(event.xdata) + ", " + str(event.ydata))
            j, i = int(event.xdata), int(event.ydata)

            if self.img[i,j] >= 0 or self.img[i,j] < -2:
                self.console.log("Data point not in training or testing set")
                return
            x, y = find_data_point(i, j)
            self.plot_data_point(x, y)
            
            
        def find_data_point(i, j):
            if self.img[i,j] == -1:
                k = self.train_mapper[f"{i} {j}"]
                #self.console.log("Found point in train set")
                return self.X_train[k], self.Y_train[k]
            
            if self.img[i,j] == -2:
                k = self.test_mapper[f"{i} {j}"]
                #self.console.log("Found point in test set")
                return self.X_test[k], self.Y_test[k]        
            
            point = self.spaceNd[i,j]
            #self.console.log("Found point in value region")
            return point, None
            
            
        self.fig.canvas.mpl_connect('motion_notify_event', display_annotation)           
        self.fig.canvas.mpl_connect('button_press_event', onclick)
       
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
        # clear the figure
        #self.dbm_fig.clf()   
        self.fig_agg.get_tk_widget().forget()
             
        # update the figure
        self.dbm_ax.set_title("Decision Boundary Mapper")
        self.proj_errs_ax.set_title("Projection Errors")
        self.inv_proj_errs_ax.set_title("Inverse Projection Errors")
        
        self.dbm_ax.imshow(self.color_img)
        self.inv_proj_errs_ax.imshow(self.inverse_projection_errors)
        
        if not self.proj_errs_computed:
            self.proj_errs_warning_text = self.proj_errs_ax.text(0.5, 0.5, "To compute projection errors you should click the button.\nThis might take about 30 seconds", transform=self.proj_errs_ax.transAxes, ha="center")
        
        
        # draw the figure to the canvas
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig)    
    
    def _draw_projection_errors_img_(self, projection_errors):
        # clear the figure
        self.fig_agg.get_tk_widget().forget()
        
        # update the figure        
        self.proj_errs_warning_text.set_visible(False)
        ax_img = self.proj_errs_ax.imshow(projection_errors)
                
        self.fig.colorbar(ax_img, ax=self.proj_errs_ax)        
        self.fig.set_size_inches(1, 1)        
        
        # draw the figure to the canvas
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig) 

    def _set_loading_proj_errs_state_(self):
        self.window['-COMPUTE PROJECTION ERRORS-'].update(visible=False, disabled=True)        
        self.proj_errs_warning_text.set_text("Computing projection errors...\nPlease wait...")
        self.fig_agg.get_tk_widget().forget()
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig) 

    def handle_compute_projection_errors_event(self, event, values):
        self._set_loading_proj_errs_state_()
        projection_errors = self.dbm_model.generate_projection_errors()
        self.proj_errs_computed = True
        self._draw_projection_errors_img_(projection_errors)
