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
from PIL import Image
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

from .. import Logger

COLORS_MAPPER = {
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
    COLORS_MAPPER[i] = [i/200,i/200,i/200]

class DBMPlotter:
    def __init__ (self, img, img_confidence, 
                  img_projection_errors, img_inverse_projection_errors, 
                  encoded_train, encoded_test, X_train, Y_train, X_test, Y_test, 
                  spaceNd, logger=None):
        
        """Decision Boundary Mapper Plotter.
           Plots the decision boundary of a trained model.
           Along with the projection errors and the inverse projection errors.
           
           Args:
                img (numpy.ndarray): The decision boundary image.
                img_confidence (numpy.ndarray): The confidence of the decision boundary image.
                img_projection_errors (numpy.ndarray): The projection errors of the decision boundary image.
                img_inverse_projection_errors (numpy.ndarray): The inverse projection errors of the decision boundary image.
                encoded_train (numpy.ndarray): The encoded train data.
                encoded_test (numpy.ndarray): The encoded test data.
                X_train (numpy.ndarray): The original train data.
                X_test (numpy.ndarray): The original test data.
                Y_train (numpy.ndarray): The original train labels.
                Y_test (numpy.ndarray): The original test labels.
                spaceNd (numpy.ndarray): The Nd coordinates of the decision boundary.
                logger (Logger): The logger to use. Defaults to None. (prints to console)
        """
        
        if logger is None:
            self.console = Logger(name="DBMPlotter")
        else:
            self.console = logger
        
        self.img = img
        self.img_confidence = img_confidence
        self.img_projection_errors = img_projection_errors
        self.img_inverse_projection_errors = img_inverse_projection_errors
        self.encoded_train = encoded_train
        self.encoded_test = encoded_test
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.spaceNd = spaceNd.reshape(spaceNd.shape[0], spaceNd.shape[1], X_train.shape[1], X_train.shape[2])
        self.color_img, self.legend = self._build_2D_image_(img, img_confidence)
        self.train_mapper, self.test_mapper = self._generate_encoded_mapping_()
        self._initialize_plot_()
    
    def _initialize_plot_(self): 
        """Initializes the plotter."""   
        plt.close("all")
        
        self.fig = plt.figure(figsize = (15, 10))
        self.ax = plt.subplot(1,2,1)
        self.ax_proj_errs = plt.subplot(2, 2, 2) 
        self.ax_inv_proj_errors = plt.subplot(2, 2, 4)
        
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax_proj_errs.xaxis.set_visible(False)
        self.ax_proj_errs.yaxis.set_visible(False)
        self.ax_inv_proj_errors.xaxis.set_visible(False)
        self.ax_inv_proj_errors.yaxis.set_visible(False)
        self._build_annotation_mapper_()
    
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

    def _build_annotation_mapper_(self):
        """ Builds the annotation mapper.
            This is used to display the data point label when hovering over the decision boundary.
        """
        image = OffsetImage(self.X_train[0], zoom=2, cmap="gray")
        label = TextArea("Data point label: None")

        xybox=(50., 50.)
        annImage = AnnotationBbox(image, (0,0), xybox=xybox, xycoords='data',
                boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))
        annImage_ax1 = AnnotationBbox(image, (0,0), xybox=xybox, xycoords='data',
                boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))
        annImage_ax2 = AnnotationBbox(image, (0,0), xybox=xybox, xycoords='data',
                boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))

        annLabels = AnnotationBbox(label, (0,0), xybox=xybox, xycoords='data',
                boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

        self.ax.add_artist(annImage)
        self.ax.add_artist(annLabels)
        self.ax_inv_proj_errors.add_artist(annImage_ax1)
        self.ax_proj_errs.add_artist(annImage_ax2)
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
        
    def show(self, title="Decision Boundary Mapper"):
        """Plots the decision boundary mapper.

        Args:
            title (str, optional): Defaults to "Decision Boundary Mapper".
        """
        self._initialize_plot_()
        self.ax.imshow(self.color_img)
        self.ax.set_title(title)
        self.ax.legend(handles=self.legend, bbox_to_anchor=(0, 1), loc=1, borderaxespad=0. )
        
        ax_img1 = self.ax_proj_errs.imshow(self.img_projection_errors)
        self.fig.colorbar(ax_img1, ax=self.ax_proj_errs)
        self.ax_proj_errs.set_title("Projection Errors")
        
        ax_img2 = self.ax_inv_proj_errors.imshow(self.img_inverse_projection_errors)
        self.fig.colorbar(ax_img2, ax=self.ax_inv_proj_errors)
        self.ax_inv_proj_errors.set_title("Inverse Projection Errors")
        
        self.fig.show()
        plt.show()