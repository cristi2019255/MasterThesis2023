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
import threading
import tensorflow as tf
from math import sqrt
from datetime import datetime
import shutil
from sklearn.metrics import cohen_kappa_score
from matplotlib.patches import Patch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

from .. import Logger, LoggerInterface
from ..DBM import DBM, SDBM 
from ..utils import TRAIN_DATA_POINT_MARKER, TEST_DATA_POINT_MARKER, TRAIN_2D_FILE_NAME, TEST_2D_FILE_NAME, INVERSE_PROJECTION_ERRORS_FILE, PROJECTION_ERRORS_INTERPOLATED_FILE, PROJECTION_ERRORS_INVERSE_PROJECTION_FILE, get_latest_created_file_from_folder, run_timer

CLASSIFIER_PERFORMANCE_HISTORY_FILE = "classifier_performance.log"
CLASSIFIER_REFIT_FOLDER = "refit_classifier"
CLASSIFIER_STACKED_FOLDER = "stacked_classifier"
CLASSIFIER_STACKED_LABELS_FILE = "classifier_old_labels.npy"
CLASSIFIER_STACKED_LABELS_CHANGES_FILE = "classifier_old_labels_changes.npy"
CLASSIFIER_STACKED_BOUNDARY_MAP_FILE = "classifier_old_boundary_map.npy"
CLASSIFIER_STACKED_CONFIDENCE_MAP_FILE = "classifier_old_boundary_map_confidence.npy"

PLOT_SNAPSHOTS_FOLDER = "plot_snapshots"

LABELS_CHANGES_FILE = "label_changes.json"
LABELS_RESULT_FILE = "labels_result.npy"

USER_ALLOWED_INTERACTION_ITERATIONS = 5

TIMER_SOUND_FILE_PATH = os.path.join(os.path.dirname(__file__), "assets", "warning-sound.mp3")
TIMER_DURATION = int(3 * 60) # seconds
TIMER_WARNING_INTERVAL = 30 # seconds

EPOCHS_FOR_REFIT = 20
EPOCHS_FOR_REFIT_RANGE = (1, 100)

class DBMPlotterController:
    def __init__(self,
                logger,
                dbm_model: DBM | SDBM,
                img: np.ndarray, 
                img_confidence: np.ndarray,
                X_train: np.ndarray, 
                Y_train: np.ndarray,
                X_test: np.ndarray, 
                Y_test: np.ndarray,
                X_train_2d: np.ndarray | None, 
                X_test_2d: np.ndarray | None, 
                encoded_train: np.ndarray, 
                encoded_test: np.ndarray,
                save_folder: str,
                projection_technique: str | None,
                gui,
                X_train_latent: np.ndarray | None = None,
                X_test_latent: np.ndarray | None = None,
                helper_decoder: tf.keras.Model | None = None
                ):
        """[summary] DBMPlotterController is the controller for the DBMPlotterGUI that will perform all the non GUI related computations.


        Args:
            dbm_model (DBM | SDBM): The DBM model that will be used to generate the decision boundary map.
            img (np.ndarray): The decision boundary map image.
            img_confidence (np.ndarray): The decision boundary map confidence image.
            X_train (np.ndarray): The training data.
            Y_train (np.ndarray): The training labels.
            X_test (np.ndarray): The test data.
            Y_test (np.ndarray): The test labels.
            X_train_2d (np.ndarray | None): The 2D embedding space of the training data, if provided it will be used to faster load the 2D plot. Defaults to None 
            X_test_2d (np.ndarray | None): The 2D embedding space of the testing data, if provided it will be used to faster load the 2D plot. Defaults to None
            encoded_train (np.ndarray): Positions of the training data points in the 2D embedding space.
            encoded_test (np.ndarray): Positions of the test data points in the 2D embedding space.
            save_folder (string): The folder where all the DBM model related files will be saved.
            projection_technique (string, optional): The projection technique the user wants to use if DBM is used as dbm_model. Defaults to None.
            gui (GUI, optional): The GUI that started the DBMPlotterGUI if any. Defaults to None.
            -------------------------------------------------------------------------------------------------------------------------------
            helper_decoder (tf.keras.Model, optional): The feature extractor helper decoder. When provided together with helper_encoder it will be used to reduce the dimensionality of the data. Defaults to None.
            X_train_latent (np.ndarray | None, optional) The features of the X_train when dimensionality reduction is needed for X_train. Defaults to None,
            X_test_latent (np.ndarray | None, optional) The features of the X_test when dimensionality reduction is needed for X_test. Defaults to None,
          """
        
        if logger is None:
            self.console = Logger(name="DBMPlotterController")
        else:
            self.console = logger
        
       
        if helper_decoder is not None and X_train_latent is not None and X_test_latent is not None:
            self.console.log("Helper feature extractor was provided, data features will be used accordingly...")
        
        self.helper_decoder = helper_decoder
        self.X_train_latent = X_train_latent
        self.X_test_latent = X_test_latent
        

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
        self.gui = gui # reference to the gui that uses the controller
        self.user_allowed_interaction_iterations = USER_ALLOWED_INTERACTION_ITERATIONS
        self.show_tooltip_for_dataset_only = False
        
        # -------------- Create folder structure --------------------
        folders_to_create = [
            CLASSIFIER_REFIT_FOLDER,
            CLASSIFIER_STACKED_FOLDER,
            PLOT_SNAPSHOTS_FOLDER
        ]
        for folder in folders_to_create:
            dir = os.path.join(self.save_folder, folder)
            if not os.path.exists(dir):
                os.makedirs(dir)
        
        # ----------------------------------------------------------------
        self.stop_timer_event = threading.Event()
        
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
        # stop the timer if there is any
        self.stop_timer()
        
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
        for value in colors_mapper.keys():
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

        times, accuracies, losses, kappas = [], [], [], []
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                
                information = line.split(" || ")
                info_dict = eval(information[1])
                accuracy, loss, kappa_score = info_dict["Accuracy"], info_dict["Loss"], info_dict["Kappa"]
                time = information[0].split(" ")[1]
                times.append(time)
                accuracies.append(float(accuracy))
                losses.append(float(loss))
                kappas.append(float(kappa_score))
        return times, accuracies, losses, kappas
    
    def compute_classifier_metrics(self):
        self.console.log("Evaluating classifier...")
    
        X_test = self.X_test if self.X_test_latent is None else self.X_test_latent
       
        loss, accuracy = self.dbm_model.classifier.evaluate(X_test, self.Y_test, verbose=0)
        Y_pred = self.dbm_model.classifier.predict(X_test, verbose=0).argmax(axis=-1)
        kappa_score = cohen_kappa_score(self.Y_test, Y_pred)
        self.console.log(f"Classifier Accuracy: {(100 * accuracy):.2f}%  Loss: {loss:.4f} Kappa: {kappa_score:.4f}")

        path = os.path.join(self.save_folder, CLASSIFIER_PERFORMANCE_HISTORY_FILE)

        with open(path, "a") as f:
            time = datetime.now().strftime("%D %H:%M:%S")
            message = "{" + f"'Accuracy': {(100 * accuracy):.2f}, 'Loss': {loss:.4f}, 'Kappa': {kappa_score:.4f}" + "}"
            f.write(f"{time} || {message}\n")
        return accuracy, loss, kappa_score
    
    def pop_classifier_evaluation(self):
        path = os.path.join(self.save_folder, CLASSIFIER_PERFORMANCE_HISTORY_FILE)
        # removing the last line
        with open(path, "r") as f:
            lines = f.readlines()
            lines = lines[:-1]
        with open(path, "w") as f:
            f.write("".join(lines))

        if len(lines) == 0:
            return None, None, None

        last_line = lines[-1].rstrip()
        info_dict = eval(last_line.split(" || ")[1])
        accuracy, loss, kappa_score = info_dict["Accuracy"], info_dict["Loss"], info_dict["Kappa"]
        return accuracy, loss, kappa_score
    
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
                
        # revoke the latest plot snapshot
        if not os.path.exists(os.path.join(self.save_folder, PLOT_SNAPSHOTS_FOLDER)):
            return
        
        plot_snapshots_folder = os.path.join(self.save_folder, PLOT_SNAPSHOTS_FOLDER)
        try:
            latest_plot_snapshot_file_path = get_latest_created_file_from_folder(plot_snapshots_folder)
            os.rename(
                      latest_plot_snapshot_file_path,
                      latest_plot_snapshot_file_path.replace(".png", "_revoked.png")
                      )
        except FileExistsError:
            self.console.error("Couldn't revoke the latest plot snapshot since it is already revoked.")
        except Exception as e:
            self.console.error(f"Error while revoking the latest plot snapshot: {str(e)}")

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
    
    def get_encoded_test_data(self):
        return self.encoded_test
    
    def get_positions_of_labels_changes(self):
        return self.positions_of_labels_changes
    
    def regenerate_boundary_map(self, Y_transformed, fast_decoding_strategy):
        X_train = self.X_train if self.X_train_latent is None else self.X_train_latent
        X_test = self.X_test if self.X_test_latent is None else self.X_test_latent
        
        if isinstance(self.dbm_model, SDBM):
            dbm_info = self.dbm_model.generate_boundary_map(
                X_train = X_train,
                Y_train = Y_transformed,
                X_test = X_test,
                Y_test = self.Y_test,
                resolution=len(self.img),
                fast_decoding_strategy=fast_decoding_strategy,
                load_folder=self.save_folder,
            )
        else:
            if self.X_train_2d is None or self.X_test_2d is None:
                self.X_train_2d, self.X_test_2d = self.load_2d_projection()
                
            dbm_info = self.dbm_model.generate_boundary_map(
                Xnd_train=X_train,
                Xnd_test=X_test,
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
        if self.user_allowed_interaction_iterations <= 0:
            message = "Max number of iterations reached! Applying labels changes is not allowed anymore!"
            self.console.error(message)
            self.updates_logger.error(message)
        
        num_changes = len(self.expert_updates_labels_mapper)
        
        if num_changes == 0:
            message = "No changes to apply!"
            self.console.error(message)
            self.updates_logger.error(message)
            return
        
        # store the changes done so far so we can restore them when needed
        with open(os.path.join(self.save_folder, CLASSIFIER_STACKED_LABELS_CHANGES_FILE), "wb") as f:
            np.save(f, self.positions_of_labels_changes)

        Y_transformed, label_changes, positions_of_labels_changes = self.transform_changes(self.Y_train, self.expert_updates_labels_mapper, self.positions_of_labels_changes)
        
        if len(label_changes) > 0.8 * len(self.Y_train):
            message = "The amount of changes can not be more than 80% of the training set in one iteration"
            self.console.error(message)
            self.updates_logger.error(message)
            return
        
        # stop the user timer
        self.stop_timer()
        
        # block the user from applying changes
        self.gui.window["-APPLY CHANGES SECTION-"].update(visible=False)
        
        self.positions_of_labels_changes = positions_of_labels_changes

        self.console.log("Saving changes to a local folder...")
        self.save_labels_changes(self.save_folder, label_changes=label_changes)

        self.updates_logger.log("Applying changes... This might take some time...")

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

        # store the plot presented when the user applies the changes
        current_time = datetime.now().strftime("%D %H:%M:%S").replace(" ", "_").replace("/", "_")
        self.gui.fig.savefig(os.path.join(self.save_folder, PLOT_SNAPSHOTS_FOLDER, f"{current_time}.png"))

        if epochs is None:
            epochs = EPOCHS_FOR_REFIT
        
        self.updates_logger.log(f"The classifier will be retrained for {epochs} epochs")

        X_train = self.X_train if self.X_train_latent is None else self.X_train_latent
        self.dbm_model.refit_classifier(X_train, Y_transformed, save_folder=save_folder, epochs=epochs)
        self.regenerate_boundary_map(Y_transformed, decoding_strategy)

        self.updates_logger.log("Changes applied successfully!")

        # decrease the number of user allowed interactions
        self.user_allowed_interaction_iterations -= 1
        
        # if user is not allowed interact block the buttons and show a informative message
        if self.user_allowed_interaction_iterations <= 0:
            # store the labels results after the usage
            with open(os.path.join(self.save_folder, LABELS_RESULT_FILE), "wb") as f:
                np.save(f, self.Y_train)
            self.gui.window["-USAGE ITERATIONS LEFT TEXT-"].update("Max number of iterations reached!\nApplying labels changes is not allowed anymore!", text_color='red')
            self.gui.window["-APPLY CHANGES-"].hide_row()
            self.gui.window["-UNDO CHANGES-"].hide_row()
            self.timer_ui.hide_row()
            self.gui.window["-APPLY CHANGES SECTION-"].update(visible=True)
            return
        
        # update the number of user allowed interactions in the gui
        self.gui.window["-USAGE ITERATIONS LEFT TEXT-"].update(f"Usage iterations left: {self.user_allowed_interaction_iterations}")
               
        # restart the timer
        self.start_timer(self.timer_ui)
        
        self.gui.window["-APPLY CHANGES SECTION-"].update(visible=True)
        
    def start_timer(self, ui_component):
        # create a new timer that will update the ui_component
        # set the stop_timer_event in order to stop the timer when needed
        self.stop_timer_event = threading.Event()
        self.timer_ui = ui_component
        self.timer_ui.update(visible=True)
        
        def timer_update_callback(msg, is_time_up):
            text_color = "red" if is_time_up else "green"
            self.timer_ui.update(msg, text_color=text_color)
            self.gui.window.refresh()
            
        threading.Thread(target=run_timer,
                        args=(
                            timer_update_callback,
                            self.stop_timer_event,
                            TIMER_DURATION,
                            TIMER_WARNING_INTERVAL,
                            TIMER_SOUND_FILE_PATH,
                            True,
                            )).start()
       
    def stop_timer(self):
        self.stop_timer_event.set()
    
    def set_dbm_model_logger(self, logger):
        self.dbm_model.console = logger
    
    def set_updates_logger(self, logger):
        self.updates_logger = logger
    
    def set_show_tooltip_for_dataset_only(self, value):
        self.show_tooltip_for_dataset_only = value
    
    def build_annotation_mapper(self, fig, ax, connect_click_event=True):
        """ Builds the annotation mapper.
            This is used to display the data point label when hovering over the decision boundary.
        """
        TOOLTIP_SIZE = 100.
        zoom = TOOLTIP_SIZE / self.X_train[0].shape[0]
        zoom = zoom if isinstance(zoom, float) else 1
        image = OffsetImage(self.X_train[0], zoom=zoom)
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
            annImage.xybox = (TOOLTIP_SIZE * ws, TOOLTIP_SIZE * hs)
            annLabels.xybox = (-50. * ws, 50. * hs)
            # make annotation box visible
            annImage.set_visible(True)
            # place it at the position of the event scatter point
            annImage.xy = (j, i)

            x_data, y_data = find_data_point(i, j)
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

            if self.show_tooltip_for_dataset_only:
                return None, None
            
            # generate the nD data point on the fly using the inverse projection
            if self.helper_decoder is not None:
                point = self.helper_decoder.predict(self.dbm_model.neural_network.decode(
                    [(i/self.img.shape[0], j/self.img.shape[1])]), verbose=0)[0]
            else:
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
        if connect_click_event:
            self.click_event_cid = fig.canvas.mpl_connect('button_press_event', onclick)
