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
import numpy as np
import json
from math import sqrt
from datetime import datetime
import shutil
from matplotlib.patches import Patch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

from .. import Logger
from ..utils import TRAIN_DATA_POINT_MARKER, TEST_DATA_POINT_MARKER, TRAIN_2D_FILE_NAME, TEST_2D_FILE_NAME, INVERSE_PROJECTION_ERRORS_FILE, PROJECTION_ERRORS_INTERPOLATED_FILE, PROJECTION_ERRORS_INVERSE_PROJECTION_FILE

CLASSIFIER_PERFORMANCE_HISTORY_FILE = "classifier_performance.log"
CLASSIFIER_REFIT_FOLDER = "refit_classifier"
CLASSIFIER_STACKED_FOLDER = "stacked_classifier"
CLASSIFIER_STACKED_LABELS_FILE = "classifier_old_labels.npy"
CLASSIFIER_STACKED_LABELS_CHANGES_FILE = "classifier_old_labels_changes.npy"
CLASSIFIER_STACKED_BOUNDARY_MAP_FILE = "classifier_old_boundary_map.npy"
CLASSIFIER_STACKED_CONFIDENCE_MAP_FILE = "classifier_old_boundary_map_confidence.npy"

LABELS_CHANGES_FILE = "label_changes.json"

EPOCHS_FOR_REFIT = 2
EPOCHS_FOR_REFIT_RANGE = (1, 100)

class DBMPlotterController:
    def __init__(self,
                logger,
                dbm_model,
                img, img_confidence,
                X_train, Y_train,
                X_test, Y_test,
                X_train_2d, X_test_2d,
                encoded_train, encoded_test,
                save_folder,
                projection_technique,
                ):
        
        if logger is None:
            self.console = Logger(name="DBMPlotterController")
        else:
            self.console = logger
            
        # folder where to save the changes made to the data by the user
        self.save_folder = save_folder
        # projection technique used to generate the DBM
        self.projection_technique = projection_technique
        if self.projection_technique is not None:
            self.save_folder = os.path.join(self.save_folder, self.projection_technique)
        
        self.inverse_projection_errors = None
        self.projection_errors = None
        self.positions_of_labels_changes = ([], [], [])

        self.dbm_model = dbm_model
        self.img = img
        self.img_confidence = img_confidence
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train_2d = X_train_2d
        self.X_test_2d = X_test_2d
        self.encoded_train = encoded_train
        self.encoded_test = encoded_test
        self.initialize()
        
    def initialize(self):
        self.train_mapper, self.test_mapper = self.generate_encoded_mapping()
        # --------------------- Others ------------------------------
        self.expert_updates_labels_mapper = {}
        self.motion_event_cid = None
        self.click_event_cid = None
        self.key_event_cid = None
        self.release_event_cid = None
        self.current_selected_point = None
        self.current_selected_point_assigned_label = None
        self.update_labels_by_circle_select = True
        self.update_labels_circle = None
    
    def clear_resources(self):
        files_to_delete = [
            CLASSIFIER_PERFORMANCE_HISTORY_FILE,
            LABELS_CHANGES_FILE,
            CLASSIFIER_STACKED_BOUNDARY_MAP_FILE,
            CLASSIFIER_STACKED_CONFIDENCE_MAP_FILE,
            CLASSIFIER_STACKED_LABELS_FILE,
            CLASSIFIER_STACKED_LABELS_CHANGES_FILE
        ]

        for file in files_to_delete:
            file = os.path.join(self.save_folder, file)
            if os.path.exists(file):
                os.remove(file)

        folders_to_delete = [
            CLASSIFIER_STACKED_FOLDER
        ]

        for folder in folders_to_delete:
            folder = os.path.join(self.save_folder, folder)
            if os.path.exists(folder):
                shutil.rmtree(folder)

    def build_2D_image(self, colors_mapper, class_name_mapper=lambda x: str(x)):
        """Combines the img and the img_confidence into a single image.

        Args:
            img (np.ndarray): Label image.
            img_confidence (np.ndarray): Confidence image.
            colors_mapper (dict): Mapper of labels to colors.
            class_name_mapper (function, optional): Describes how to map the data labels. Defaults to lambda x:str(x).
            
        Returns:
            np.ndarray: The combined image.
            patches: The legend patches.
        """
        img = self.img
        color_img = np.zeros((img.shape[0], img.shape[1], 3))

        for i, j in np.ndindex(img.shape):
            color_img[i][j] = colors_mapper[img[i][j]]
        values = np.unique(img)

        patches = []
        for value in values:
            color = colors_mapper[value]
            if value == TRAIN_DATA_POINT_MARKER:
                label = "Original train data"
            elif value == TEST_DATA_POINT_MARKER:
                label = "Original test data"
            else:
                label = f"Value region: {class_name_mapper(value)}"

            patches.append(Patch(color=color, label=label))

        self.color_img = color_img
        self.legend = patches

        return color_img, patches

    def generate_encoded_mapping(self):
        """Generates a mapping of the encoded data to the original data.

        Returns:
            train_mapper (dict): Mapping of the encoded train data to the original train data.
            test_mapper (dict): Mapping of the encoded test data to the original test data.
        """
        train_mapper = {}
        for k in range(len(self.encoded_train)):
            [i, j, _] = self.encoded_train[k]
            train_mapper[f"{int(i)} {int(j)}"] = k

        test_mapper = {}
        for k in range(len(self.encoded_test)):
            [i, j, _] = self.encoded_test[k]
            test_mapper[f"{int(i)} {int(j)}"] = k

        return train_mapper, test_mapper
    
    def get_classifier_performance_history(self):
        path = os.path.join(self.save_folder, CLASSIFIER_PERFORMANCE_HISTORY_FILE)
        if not os.path.isfile(path):
            raise Exception("Classifier performance history file not found.")

        times, accuracies, losses = [], [], []
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                _, time, _, acc, _, loss, _, _ = line.replace("\n", "").replace("%", "").split(" ")
                times.append(time)
                accuracies.append(float(acc))
                losses.append(float(loss))
        return times, accuracies, losses
    
    def compute_classifier_metrics(self, epochs):
        self.console.log("Evaluating classifier...")
        loss, accuracy = self.dbm_model.classifier.evaluate(self.X_test, self.Y_test, verbose=0)
        self.console.log(f"Classifier Accuracy: {(100 * accuracy):.2f}%  Loss: {loss:.2f}")

        path = os.path.join(self.save_folder, CLASSIFIER_PERFORMANCE_HISTORY_FILE)

        with open(path, "a") as f:
            time = datetime.now().strftime("%D %H:%M:%S")
            message = f"Accuracy: {(100 * accuracy):.2f}% Loss: {loss:.2f} Epochs: {epochs}"
            f.write(f"{time} {message}\n")
        return accuracy, loss
    
    def pop_classifier_evaluation(self):
        path = os.path.join(self.save_folder, CLASSIFIER_PERFORMANCE_HISTORY_FILE)
        # removing the last line
        with open(path, "r") as f:
            lines = f.readlines()
            lines = lines[:-1]
        with open(path, "w") as f:
            f.write("".join(lines))

        if len(lines) == 0:
            return None, None

        last_line = lines[-1].replace("\n", "")
        accuracy, loss = float(last_line.split("Accuracy: ")[1].split("%")[0]), float(last_line.split("Loss: ")[1])
        return accuracy, loss
    
    def load_2d_projection(self):
        if os.path.exists(os.path.join(self.save_folder, TRAIN_2D_FILE_NAME)) and os.path.exists(os.path.join(self.save_folder, TEST_2D_FILE_NAME)):
            X2d_train = np.load(os.path.join(self.save_folder, TRAIN_2D_FILE_NAME))
            X2d_test = np.load(os.path.join(self.save_folder, TEST_2D_FILE_NAME))
            return X2d_train, X2d_test
        return None, None
    
    def transform_changes(self, Y_train, expert_updates_labels_mapper, positions_of_labels_changes):
        """
        Returns:
            position_changes (dict): a dictionary with old position as key and the new position as value
            label_changes (dict): a dictionary with old and new positions as key and the new label as value
        """
        Y = np.copy(Y_train)
        labels_changes = {}

        pos_x, pos_y, alphas = positions_of_labels_changes

        ALPHA_DECAY_ON_UPDATE = 0.05
        alphas = [alpha - ALPHA_DECAY_ON_UPDATE if alpha > ALPHA_DECAY_ON_UPDATE else ALPHA_DECAY_ON_UPDATE for alpha in alphas]

        for pos in expert_updates_labels_mapper:
            k = self.train_mapper[pos]
            pos_x = np.append(pos_x, int(pos.split(" ")[1]))
            pos_y = np.append(pos_y, int(pos.split(" ")[0]))
            alphas = np.append(alphas, 1)
            y = expert_updates_labels_mapper[pos][0]
            Y[k] = y
            labels_changes[pos] = expert_updates_labels_mapper[pos][0]

        positions_of_labels_changes = (pos_x, pos_y, alphas)

        return Y, labels_changes, positions_of_labels_changes
    
    def save_labels_changes(self, folder: str = "tmp", label_changes={}):
        if not os.path.exists(folder):
            os.path.makedirs(folder)

        with open(os.path.join(folder, LABELS_CHANGES_FILE), "a") as f:
            f.write("\n" + "-" * 50 + "\n")
            json.dump(label_changes, f, indent=2)

    def compute_inverse_projection_errors(self):
        possible_path = os.path.join(self.save_folder, INVERSE_PROJECTION_ERRORS_FILE)
        # try to get projection errors from cache first
        if os.path.exists(possible_path):
            self.inverse_projection_errors = np.load(possible_path)
        else:
            self.inverse_projection_errors = self.dbm_model.generate_inverse_projection_errors(save_folder=self.save_folder, resolution=len(self.img))
            
    def compute_projection_errors(self, type = "non_interpolated"):
        if type == "interpolated":
            use_interpolation = True
            file = PROJECTION_ERRORS_INTERPOLATED_FILE
        elif type == "non_interpolated":
            use_interpolation = False
            file = PROJECTION_ERRORS_INVERSE_PROJECTION_FILE
            
        possible_path = os.path.join(self.save_folder, file)

        # try to get projection errors from cache first
        if os.path.exists(possible_path):
            self.projection_errors = np.load(possible_path)
        else:
            self.projection_errors = self.dbm_model.generate_projection_errors(use_interpolation=use_interpolation, save_folder=self.save_folder)

    def undo_changes(self):
        if not os.path.exists(os.path.join(self.save_folder, CLASSIFIER_STACKED_FOLDER)):
            raise Exception("Can not undo changes, no previous model found")

        if not os.path.exists(os.path.join(self.save_folder, CLASSIFIER_STACKED_LABELS_FILE)):
            raise Exception("Can not undo changes, no previous labels found")

        if not os.path.exists(os.path.join(self.save_folder, CLASSIFIER_STACKED_BOUNDARY_MAP_FILE)):
            raise Exception("Can not undo changes, no previous boundary map found")

        if not os.path.exists(os.path.join(self.save_folder, CLASSIFIER_STACKED_CONFIDENCE_MAP_FILE)):
            raise Exception("Can not undo changes, no previous confidence map found")
        
        if not os.path.exists(os.path.join(self.save_folder, CLASSIFIER_STACKED_LABELS_CHANGES_FILE)):
            raise Exception("Can not undo changes, no previous labels changes found")
        
        if not os.path.exists(os.path.join(self.save_folder, CLASSIFIER_STACKED_FOLDER)):
            raise Exception("Can not undo changes, no previous model found")

        self.dbm_model.load_classifier(os.path.join(self.save_folder, CLASSIFIER_STACKED_FOLDER))

        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_LABELS_FILE), "rb") as f:
            self.Y_train = np.load(f)
        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_BOUNDARY_MAP_FILE), "rb") as f:
            self.img = np.load(f)
        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_CONFIDENCE_MAP_FILE), "rb") as f:
            self.img_confidence = np.load(f)
        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_LABELS_CHANGES_FILE), "rb") as f:
            self.positions_of_labels_changes = np.load(f)
            
        files_to_delete = [
            CLASSIFIER_STACKED_LABELS_FILE,
            CLASSIFIER_STACKED_BOUNDARY_MAP_FILE,
            CLASSIFIER_STACKED_CONFIDENCE_MAP_FILE,
            CLASSIFIER_STACKED_LABELS_CHANGES_FILE
        ]

        for file in files_to_delete:
            file = os.path.join(self.save_folder, file)
            if os.path.exists(file):
                os.remove(file)


    def mix_image(self, show_color_map, show_confidence, show_inverse_projection_errors, show_projection_errors):
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
        return img
    
    def get_encoded_train_data(self):
        return self.encoded_train
    
    def get_positions_of_labels_changes(self):
        return self.positions_of_labels_changes
    
    def regenerate_boundary_map(self, Y_transformed, fast_decoding_strategy):
        if self.projection_technique is None:
            dbm_info = self.dbm_model.generate_boundary_map(
                self.X_train,
                Y_transformed,
                self.X_test,
                self.Y_test,
                resolution=len(self.img),
                fast_decoding_strategy=fast_decoding_strategy,
                load_folder=self.save_folder
            )
        else:
            if self.X_train_2d is None or self.X_test_2d is None:
                self.X_train_2d, self.X_test_2d = self.load_2d_projection()
            dbm_info = self.dbm_model.generate_boundary_map(
                Xnd_train=self.X_train,
                Xnd_test=self.X_test,
                X2d_train=self.X_train_2d,
                X2d_test=self.X_test_2d,
                resolution=len(self.img),
                fast_decoding_strategy=fast_decoding_strategy,
                load_folder=self.save_folder,
                projection=self.projection_technique
            )
        
        img, img_confidence, encoded_train, encoded_test = dbm_info
        
        self.img = img
        self.img_confidence = img_confidence
        self.encoded_train = encoded_train
        self.encoded_test = encoded_test
        self.Y_train = Y_transformed
        self.initialize()
        
    def apply_labels_changes(self, decoding_strategy, epochs = None):
        num_changes = len(self.expert_updates_labels_mapper)
        if num_changes == 0:
            self.console.error("No changes to apply")
            return
        if num_changes < 10:
            self.console.error("Less than 10 changes to apply, please apply more changes")
            return

        # store the changes done so far so we can restore them when needed
        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_LABELS_CHANGES_FILE), "wb") as f:
            np.save(f, self.positions_of_labels_changes)

        self.console.log("Transforming changes...")
        Y_transformed, label_changes, positions_of_labels_changes = self.transform_changes(self.Y_train, self.expert_updates_labels_mapper, self.positions_of_labels_changes)
        self.positions_of_labels_changes = positions_of_labels_changes

        self.console.log("Saving changes to a local folder...")
        self.save_labels_changes(self.save_folder, label_changes=label_changes)

        self.updates_logger.log("Applying changes... This might take a couple of seconds...")

        save_folder = os.path.join(self.save_folder, CLASSIFIER_REFIT_FOLDER)

        # store the old model so we can restore it when needed
        self.dbm_model.save_classifier(save_folder=os.path.join(self.save_folder, CLASSIFIER_STACKED_FOLDER))
        # store the old labels so we can restore them when needed
        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_LABELS_FILE), "wb") as f:
            np.save(f, self.Y_train)

        # store the old decision boundary map and confidence map so we can restore them when needed
        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_BOUNDARY_MAP_FILE), "wb") as f:
            np.save(f, self.img)
        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_CONFIDENCE_MAP_FILE), "wb") as f:
            np.save(f, self.img_confidence)

        if epochs is None:
            epochs = EPOCHS_FOR_REFIT
        
        self.updates_logger.log(f"The classifier will be retrained for {epochs} epochs")

        self.dbm_model.refit_classifier(self.X_train, Y_transformed, save_folder=save_folder, epochs=epochs)
        self.regenerate_boundary_map(Y_transformed, decoding_strategy)
        
    def set_dbm_model_logger(self, logger):
        self.dbm_model.console = logger
    
    def set_updates_logger(self, logger):
        self.updates_logger = logger
    
    def build_annotation_mapper(self, fig, ax):
        """ Builds the annotation mapper.
            This is used to display the data point label when hovering over the decision boundary.
        """
        image = OffsetImage(self.X_train[0], zoom=2, cmap="gray")
        label = TextArea("Data point label: None")

        annImage = AnnotationBbox(image, (0, 0), xybox=(50., 50.), xycoords='data', boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))
        annLabels = AnnotationBbox(label, (0, 0), xybox=(50., 50.), xycoords='data', boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

        ax.add_artist(annImage)
        ax.add_artist(annLabels)
        annImage.set_visible(False)
        annLabels.set_visible(False)

        fig_width, fig_height = fig.get_size_inches() * fig.dpi

        def display_annotation(event):
            """Displays the annotation box when hovering over the decision boundary based on the mouse position."""
            if event.inaxes == None:
                return

            j, i = int(event.xdata), int(event.ydata)

            # change the annotation box position relative to mouse.
            ws = (event.x > fig_width/2.)*-1 + (event.x <= fig_width/2.)
            hs = (event.y > fig_height/2.)*-1 + (event.y <= fig_height/2.)
            annImage.xybox = (50. * ws, 50. * hs)
            annLabels.xybox = (-50. * ws, 50. * hs)
            # make annotation box visible
            annImage.set_visible(True)
            # place it at the position of the event scatter point
            annImage.xy = (j, i)

            x_data, y_data = find_data_point(i, j)
            if x_data is not None:
                annImage.set_visible(True)
                resize_factor = int(len(x_data) / 50 + 1)
                annImage.xybox = (50. * ws * resize_factor, 50. * hs * resize_factor)
                image.set_data(x_data)
            else:
                annImage.set_visible(False)

            if y_data is not None:
                annLabels.set_visible(True)
                annLabels.xy = (j, i)
                label.set_text(f"{y_data}")
            else:
                annLabels.set_visible(False)

            fig.canvas.draw_idle()

        def find_data_point(i, j):
            # search for the data point in the encoded train data
            if self.img[i][j] == TRAIN_DATA_POINT_MARKER:
                k = self.train_mapper[f"{i} {j}"]
                if f"{i} {j}" in self.expert_updates_labels_mapper:
                    l = self.expert_updates_labels_mapper[f"{i} {j}"][0]
                    return self.X_train[k], f"Label {self.Y_train[k]} \nClassifier label: {int(self.encoded_train[k][2])} \nExpert label: {l}"
                return self.X_train[k], f"Label: {self.Y_train[k]} \nClassifier label: {int(self.encoded_train[k][2])}"

            # search for the data point in the encoded test data
            if self.img[i][j] == TEST_DATA_POINT_MARKER:
                k = self.test_mapper[f"{i} {j}"]
                return self.X_test[k], f"Classifier label: {int(self.encoded_test[k][2])}"

            # generate the nD data point on the fly using the inverse projection
            point = self.dbm_model.neural_network.decode([(i/self.img.shape[0], j/self.img.shape[1])])[0]
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

            if self.img[i][j] != TRAIN_DATA_POINT_MARKER:
                self.console.log("Data point not in training set")
                return

            # check if clicked on a data point that already was updated, then remove the update
            if f"{i} {j}" in self.expert_updates_labels_mapper:
                (_, point) = self.expert_updates_labels_mapper[f"{i} {j}"]
                point.remove()
                del self.expert_updates_labels_mapper[f"{i} {j}"]
                fig.canvas.draw_idle()
                self.updates_logger.log(f"Removed updates for point: ({j}, {i})")
                return

            # if clicked on a data point that was not updated, then add the update
            self.current_selected_point = ax.plot(j, i, 'bo', markersize=5)[0]

            # disable annotations on hover
            # if self.motion_event_cid is not None:
            #    fig.canvas.mpl_disconnect(self.motion_event_cid)
            # disable on click event
            if self.click_event_cid is not None:
                fig.canvas.mpl_disconnect(self.click_event_cid)
                self.click_event_cid = None

            # enable key press events
            self.key_event_cid = fig.canvas.mpl_connect('key_press_event', onkey)

            fig.canvas.draw_idle()

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
                    self.current_selected_point = ax.plot(x, y, 'b^')[0]
                    if self.current_selected_point_assigned_label is not None:
                        self.updates_logger.log(f"Assigned label: {self.current_selected_point_assigned_label} to point: ({x[0]}, {y[0]})")
                        self.expert_updates_labels_mapper[f"{y[0]} {x[0]}"] = (self.current_selected_point_assigned_label, self.current_selected_point)

                self.current_selected_point = None
                self.current_selected_point_assigned_label = None

                self.motion_event_cid = fig.canvas.mpl_connect('motion_notify_event', display_annotation)
                self.click_event_cid = fig.canvas.mpl_connect('button_press_event', onclick)
                fig.canvas.mpl_disconnect(self.key_event_cid)
                fig.canvas.draw_idle()
                return

        def onclick_circle_strategy(event):
            self.console.log("Clicked on: " + str(event.xdata) + ", " + str(event.ydata))
            x, y = int(event.xdata), int(event.ydata)
            self.current_selected_point = (x, y)
            self.release_event_cid = fig.canvas.mpl_connect('button_release_event', onrelease_circle_strategy)

        def onrelease_circle_strategy(event):
            if event.inaxes == None:
                return
            self.console.log("Released on: " + str(event.xdata) + ", " + str(event.ydata))
            (x0, y0) = self.current_selected_point
            x1, y1 = int(event.xdata), int(event.ydata)
            (cx, cy) = (x0 + (x1 - x0)/2, y0 + (y1 - y0)/2)  # circle center
            r = sqrt(((x1 - x0)/2)**2 + ((y1 - y0)/2)**2)   # circle radius
            self.update_labels_circle = Circle((cx, cy), r, color='black', fill=False)
            ax.add_artist(self.update_labels_circle)
            self.key_event_cid = fig.canvas.mpl_connect('key_press_event', onkey_circle_strategy)
            if self.release_event_cid is not None:
                fig.canvas.mpl_disconnect(self.release_event_cid)
            fig.canvas.draw_idle()

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
                positions = find_points_in_circle((x, y), r)
                for pos in positions:
                    point = ax.plot(pos[0], pos[1], 'b^')[0]
                    self.expert_updates_labels_mapper[f"{pos[1]} {pos[0]}"] = (self.current_selected_point_assigned_label, point)

            if event.key == "backspace":
                self.updates_logger.log(f"Removing changes in circle: center ({x},{y}), radius ({r})")
                positions = find_points_in_circle((x, y), r)
                for pos in positions:
                    if f"{pos[1]} {pos[0]}" in self.expert_updates_labels_mapper:
                        self.expert_updates_labels_mapper[f"{pos[1]} {pos[0]}"][1].remove()
                        del self.expert_updates_labels_mapper[f"{pos[1]} {pos[0]}"]

            self.update_labels_circle.remove()
            self.update_labels_circle = None
            fig.canvas.mpl_disconnect(self.key_event_cid)
            fig.canvas.draw_idle()

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
                    if (self.img[y, x] == TRAIN_DATA_POINT_MARKER) and ((x - cx)**2 + (y - cy)**2 <= circle_radius**2):
                        positions.append((x, y))
            return positions

        self.motion_event_cid = fig.canvas.mpl_connect('motion_notify_event', display_annotation)
        self.click_event_cid = fig.canvas.mpl_connect('button_press_event', onclick)
