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

import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import matplotlib.figure as figure
import matplotlib.pyplot as plt
from PIL import Image
import PySimpleGUI as sg
import matplotlib
from matplotlib import cm

from .. import Logger, LoggerGUI, FAST_DBM_STRATEGIES
from ..DBM import DBM, SDBM
from .DBMPlotterController import DBMPlotterController
from .DBMPlotterController import EPOCHS_FOR_REFIT, EPOCHS_FOR_REFIT_RANGE, USER_ALLOWED_INTERACTION_ITERATIONS
from ..utils import TRAIN_DATA_POINT_MARKER, TEST_DATA_POINT_MARKER, BLACK_COLOR, WHITE_COLOR, RED_COLOR, GREEN_COLOR, YELLOW_COLOR, RIGHTS_MESSAGE_1, RIGHTS_MESSAGE_2, APP_PRIMARY_COLOR, APP_FONT

plt.switch_backend('agg')
matplotlib.use("TkAgg")

def draw_figure_to_canvas(canvas, figure, canvas_toolbar=None):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar is not None and canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()

    figure.set_dpi(100)
    figure.set_size_inches(1, 1)

    figure_canvas_agg = FigureCanvasTkAgg(figure, master=canvas)
    figure_canvas_agg.draw()

    if canvas_toolbar is not None:
        toolbar = NavigationToolbar2Tk(figure_canvas_agg, canvas_toolbar)
        toolbar.update()

    canvas_widget = figure_canvas_agg.get_tk_widget()
    
    canvas_widget.pack(side='top', fill='both', expand=True)
    canvas_widget.place(relx=0.5, rely=0.5, anchor='center', relheight=1, relwidth=1)
     
    return figure_canvas_agg


def generate_color_mapper(num_classes):
   
    colors = cm.tab10(np.linspace(0, 1, num_classes))
    colors_mapper = {
        # setting original test data to black
        TEST_DATA_POINT_MARKER: [0, 0, 0],
        # setting original train data to white
        TRAIN_DATA_POINT_MARKER: [1, 1, 1],
    }
    # setting the rest of the colors
    for i in range(num_classes):
        colors_mapper[i] = colors[i][:3]
    return colors_mapper



TITLE = "Decision Boundary Map & Errors"
WINDOW_SIZE = (1650, 1000)
INFORMATION_CONTROLS_MESSAGE = "To change label(s) of a data point(s) first click on the start usage button.\nThen click on the data point, or select the data point by including them into a circle.\nPress any digit key to indicate the new label. Press 'Enter' to confirm the new label. \nPress 'Esc' to cancel the action. To remove a change just click on the data point.\nPress 'Apply Changes' to update the model."
DBM_WINDOW_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "dbm_plotter_icon_b64.txt")

class DBMPlotterGUI:
    
    def __init__(self,
                 dbm_model: DBM | SDBM,
                 img: np.ndarray,
                 img_confidence: np.ndarray,
                 X_train: np.ndarray, 
                 Y_train: np.ndarray,
                 X_test: np.ndarray, 
                 Y_test: np.ndarray,
                 encoded_train: np.ndarray, 
                 encoded_test: np.ndarray,
                 save_folder: str,
                 X_train_2d: np.ndarray | None = None,
                 X_test_2d: np.ndarray | None = None,
                 projection_technique: str | None=None,
                 logger=None,
                 main_gui=None,
                 class_name_mapper= lambda x: str(x),
                 color_mapper=generate_color_mapper(10),
                 helper_decoder=None,
                 X_train_latent: np.ndarray | None = None,
                 X_test_latent: np.ndarray | None = None,
                 ):
        """[summary] DBMPlotterGUI is a GUI that allows the user to visualize the decision boundary map and the errors of the DBM model.
        It also allows the user to change the labels of the data points and see the impact of the changes on the model.

        Args:
            dbm_model (DBM | SDBM): The DBM model that will be used to generate the decision boundary map.
            img (np.ndarray): The decision boundary map image.
            img_confidence (np.ndarray): The decision boundary map confidence image.
            X_train (np.ndarray): The training data.
            Y_train (np.ndarray): The training labels.
            X_test (np.ndarray): The test data.
            Y_test (np.ndarray): The test labels.
            encoded_train (np.ndarray): Positions of the training data points in the 2D embedding space.
            encoded_test (np.ndarray): Positions of the test data points in the 2D embedding space.
            save_folder (string): The folder where all the DBM model related files will be saved.
            X_train_2d (np.ndarray | None): The 2D embedding space of the training data, if provided it will be used to faster load the 2D plot. Defaults to None 
            X_test_2d (np.ndarray | None): The 2D embedding space of the testing data, if provided it will be used to faster load the 2D plot. Defaults to None
            projection_technique (string, optional): The projection technique the user wants to use if DBM is used as dbm_model. Defaults to None.
            logger (Logger, optional): The logger which is meant for logging the info messages. Defaults to None.
            main_gui (GUI, optional): The GUI that started the DBMPlotterGUI if any. Defaults to None.
            class_name_mapper (function, optional): The function which is meant for mapping class names to their corresponding values. Defaults to lambda x -> str(x).
            color_mapper (dict, optional): The color mapper dictionary. Should include keys: TEST_DATA_POINT_MARKER = -2, TRAIN_DATA_POINT_MARKER = -1, 0, ..., num_classes. Each value should be an array of 3 numbers in range 0-1
            -------------------------------------------------------------------------------------------------------------------------------
            helper_decoder (tf.keras.Model, optional): The feature extractor helper decoder. When provided together with helper_encoder it will be used to reduce the dimensionality of the data. Defaults to None.
            X_train_latent (np.ndarray | None, optional) The features of the X_train when dimensionality reduction is needed for X_train. Defaults to None,
            X_test_latent (np.ndarray | None, optional) The features of the X_test when dimensionality reduction is needed for X_test. Defaults to None,
        """
        
        self.main_gui = main_gui  # reference to main window
        self.class_name_mapper = class_name_mapper
        if logger is None:
            self.console = Logger(name="DBMPlotterGUI")
        else:
            self.console = logger
        
        self.controller = DBMPlotterController(logger,
                                                dbm_model,
                                                img, 
                                                img_confidence,
                                                X_train, 
                                                Y_train,
                                                X_test,
                                                Y_test,
                                                X_train_2d, 
                                                X_test_2d,
                                                encoded_train, 
                                                encoded_test,
                                                save_folder,
                                                projection_technique,
                                                gui=self,
                                                X_train_latent=X_train_latent,
                                                X_test_latent=X_test_latent,
                                                helper_decoder=helper_decoder,
                                                )
        
        num_classes = len(np.unique(np.concatenate((Y_train, Y_test))))
        try:
            assert(color_mapper is dict)
            assert(TEST_DATA_POINT_MARKER in color_mapper.keys())
            assert(TRAIN_DATA_POINT_MARKER in color_mapper.keys())
            assert(len(color_mapper) == num_classes + 2)
            assert (all(len(color) == 3 and channel <= 1 and channel >=0 for channel in color for color in color_mapper.values()))
            self.colors_mapper = color_mapper
        except AssertionError:
            if color_mapper is not None:
                self.console.log("The provided color mapper does not match the requirements, falling back to the default color mapper...")
            self.colors_mapper = generate_color_mapper(num_classes)
            
        self.initialize_plots(connect_click_event = False)
        
    def initialize_plots(self, connect_click_event = True):
        self.color_img, self.legend = self.controller.build_2D_image(colors_mapper=self.colors_mapper, class_name_mapper=self.class_name_mapper)
        # --------------------- Plotter related ---------------------
        self.classifier_performance_fig, self.classifier_performance_ax = self._build_plot_()
        self.fig, self.ax = self._build_plot_()
        self.controller.build_annotation_mapper(self.fig, self.ax, connect_click_event)
        self.fig.legend(handles=self.legend, borderaxespad=0.)

    def _initialize_gui_(self):
        # --------------------- GUI related ---------------------
        self.window = self._build_GUI_()
        self.classifier_performance_canvas, self.classifier_performance_fig_agg = self._build_canvas_(self.classifier_performance_fig, key="-CLASSIFIER PERFORMANCE CANVAS-")
        self.canvas, self.fig_agg, self.canvas_controls = self._build_canvas_(self.fig, key="-DBM CANVAS-", controls_key="-CONTROLS CANVAS-")

        # --------------------- Classifier related ---------------------
        self.compute_classifier_metrics()

        self.draw_dbm_img()
        # --------------------- GUI related ---------------------
        self.updates_logger = LoggerGUI(name="Updates logger", output=self.window["-LOGGER-"], update_callback=self.window.refresh)
        self.controller.set_dbm_model_logger(self.updates_logger)
        self.controller.set_updates_logger(self.updates_logger)

    def _get_GUI_layout_(self):
        buttons_proj_errs = []
        buttons_inv_proj_errs = []
        computed_projection_errors = self.controller.projection_errors is not None
        computed_inverse_projection_errors = self.controller.inverse_projection_errors is not None
        if not computed_projection_errors:
            buttons_proj_errs = [
                [sg.Button('Compute Projection Errors (interpolation)', font=APP_FONT, expand_x=True, key="-COMPUTE PROJECTION ERRORS INTERPOLATION-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR))],
                [sg.Button('Compute Projection Errors (inverse projection)', font=APP_FONT, expand_x=True, key="-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR))]
            ]
        if not computed_inverse_projection_errors:
            buttons_inv_proj_errs = [
                sg.Button('Compute Inverse Projection Errors', font=APP_FONT, expand_x=True, key="-COMPUTE INVERSE PROJECTION ERRORS-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR))
            ]

        layout = [
            [
                sg.Column([
                    [sg.Canvas(key='-DBM CANVAS-', size=(200, 200), expand_x=True, expand_y=True, pad=(0, 0))],
                    [sg.Canvas(key='-CONTROLS CANVAS-', expand_x=True, pad=(0, 0))],
                ], pad=(0, 0), expand_x=True, expand_y=True),
                sg.VSeparator(),
                sg.Column([
                    [sg.Canvas(key="-CLASSIFIER PERFORMANCE CANVAS-", size=(200, 200), expand_y=True, expand_x=True)],
                    [
                        sg.Text("Classifier accuracy: ", font=APP_FONT, expand_x=True, key="-CLASSIFIER ACCURACY-"),
                    ],
                    [ sg.HSeparator() ],
                    [
                        sg.Checkbox("Show tooltip for points outside of the dataset", default=True, key="-SHOW TOOLTIP OUTSIDE DATASET-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0,0)),
                    ],
                    [
                        sg.Checkbox("Change labels by selecting with circle", default=True, key="-CIRCLE SELECTING LABELS-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0, 0)),
                    ],
                    [
                        sg.Checkbox("Show labels changes", default=False, key="-SHOW LABELS CHANGES-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0, 0)),
                    ],
                    [
                        sg.Checkbox("Show dbm color map", default=True, key="-SHOW DBM COLOR MAP-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0, 0)),
                    ],
                    [
                        sg.Checkbox("Show dbm confidence", default=True, key="-SHOW DBM CONFIDENCE-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0, 0)),
                    ],
                    [
                        sg.Checkbox("Show classifier predictions", default=False, key="-SHOW CLASSIFIER PREDICTIONS-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0, 0)),
                    ],
                    [
                        sg.Checkbox("Show data labels", default=False, key="-SHOW DATA LABELS-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0,0)),
                    ],
                    [
                        sg.Checkbox("Show inverse projection errors", default=False, key="-SHOW INVERSE PROJECTION ERRORS-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0, 0), visible=computed_inverse_projection_errors),
                    ],
                    [
                        sg.Checkbox("Show projection errors", default=False, key="-SHOW PROJECTION ERRORS-", enable_events=True, font=APP_FONT, expand_x=True, pad=(0, 0), visible=computed_projection_errors),
                    ],
                    [sg.HSeparator()],
                    [
                        sg.Text("DBM Decoding Strategy: ", font=APP_FONT),
                        sg.Combo(
                            values=list(FAST_DBM_STRATEGIES.list()),
                            default_value=FAST_DBM_STRATEGIES.NONE.value,
                            expand_x=True,
                            enable_events=True,
                            font=APP_FONT,
                            key="-DBM FAST DECODING STRATEGY-",
                            background_color=WHITE_COLOR,
                            text_color=BLACK_COLOR,
                            button_background_color=APP_PRIMARY_COLOR,
                            readonly=True,
                        ),
                    ],
                    [   
                        sg.vbottom(sg.Text(f"Number of epochs:", font=APP_FONT, key="-DBM RELABELING CLASSIFIER EPOCHS TEXT-")),
                        sg.Slider(range=EPOCHS_FOR_REFIT_RANGE, 
                               default_value=EPOCHS_FOR_REFIT, 
                               expand_x=True, 
                               enable_events=False,
                               orientation='horizontal',
                               trough_color=WHITE_COLOR,
                               font=APP_FONT,
                               tooltip="Select the number of epochs for which \nthe classifier will be retrained.",
                               key="-DBM RELABELING CLASSIFIER EPOCHS-")
                    ],
                    [
                        sg.HSeparator()
                    ],
                    [
                        sg.Column([
                            [sg.pin(
                                sg.Column([
                                    [
                                        sg.Button("Start Usage", font=APP_FONT, expand_x=True, key="-START APPLY CHANGES BTN-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR))
                                    ],
                                    [
                                        sg.HSeparator()
                                    ],
                                ], key="-START APPLY CHANGES SECTION-", visible=True, expand_x=True, expand_y=True),
                                shrink=True, expand_x=True, expand_y=True)],
                            ], pad=(0, 0), expand_x=True),  
                    ],
                    [
                        sg.Column([
                            [sg.pin(
                            sg.Column([
                                [
                                    sg.Text(f"", font=APP_FONT, key="-APPLY CHANGES TIMER TEXT-", expand_x=True, justification='center'),
                                ],
                                [
                                    sg.Text(f"Usage iterations left: {USER_ALLOWED_INTERACTION_ITERATIONS}", font=APP_FONT, key="-USAGE ITERATIONS LEFT TEXT-", expand_x=True, justification='center', text_color='green'),
                                ],
                                [
                                    sg.Button("Apply Changes", font=APP_FONT, expand_x=True, key="-APPLY CHANGES-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR)),
                                    sg.Button("Undo Changes", font=APP_FONT, expand_x=True, key="-UNDO CHANGES-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR)),
                                ],
                                [
                                    sg.Button("Pause Usage", font=APP_FONT, expand_x=True, key="-PAUSE USAGE BTN-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR)),  
                                ],
                                [
                                    sg.HSeparator()
                                ],
                            ], key="-APPLY CHANGES SECTION-", visible=False, expand_x=True, expand_y=True),
                            shrink=True, expand_x=True, expand_y=True)],
                        ], pad=(0, 0), expand_x=True),  
                    ],
                    [
                        sg.Column([
                            [sg.pin(
                                sg.Column([
                                    buttons_proj_errs[0],
                                    buttons_proj_errs[1],
                                    buttons_inv_proj_errs,
                                ], key="-PROJECTION ERRORS SECTION-", visible=True, expand_x=True, expand_y=True),
                                shrink=True, expand_x=True, expand_y=True)],
                        ], pad=(0, 0), expand_x=True),
                    ],
                    [sg.Multiline("", key="-LOGGER-", size=(40, 20),
                                          font=APP_FONT,
                                          background_color=WHITE_COLOR,
                                          text_color=BLACK_COLOR,
                                          sbar_background_color=APP_PRIMARY_COLOR,
                                          auto_size_text=True,
                                          expand_y=True, expand_x=True,
                                          enable_events=False,
                                          disabled=True)],
                    
                    [sg.Text(INFORMATION_CONTROLS_MESSAGE, font=APP_FONT, expand_x=True)],
                    [sg.Text(RIGHTS_MESSAGE_1, font=APP_FONT, expand_x=True)],
                    [sg.Text(RIGHTS_MESSAGE_2, font=APP_FONT, expand_x=True)],
                ]),
            ]
        ]

        return layout

    def _build_GUI_(self):
        window = sg.Window(title=TITLE,
                           layout=self._get_GUI_layout_(),
                           size=WINDOW_SIZE,
                           resizable=True,
                           element_justification='center',
                           )
        
        with open(DBM_WINDOW_ICON_PATH, "rb") as f:
            iconb64 = f.read()

        window.finalize()
        window.maximize()
        
        window.set_icon(pngbase64=iconb64)

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
        if self.main_gui is not None and hasattr(self.main_gui, "handle_changes_in_dbm_plotter"):
            self.main_gui.handle_changes_in_dbm_plotter()
        self.console.log("Closing the application...")

        self.controller.clear_resources()
        
        self.window.close()

    def handle_event(self, event, values):
        EVENTS = {
            "-COMPUTE PROJECTION ERRORS INTERPOLATION-": self.handle_compute_projection_errors_event,
            "-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-": self.handle_compute_projection_errors_event,
            "-COMPUTE INVERSE PROJECTION ERRORS-": self.handle_compute_inverse_projection_errors_event,
            "-APPLY CHANGES-": self.handle_apply_changes_event,
            "-SHOW TOOLTIP OUTSIDE DATASET-": self.handle_tooltip_checkbox_change_event,
            "-SHOW DBM COLOR MAP-": self.handle_checkbox_change_event,
            "-SHOW DBM CONFIDENCE-": self.handle_checkbox_change_event,
            "-SHOW INVERSE PROJECTION ERRORS-": self.handle_checkbox_change_event,
            "-SHOW PROJECTION ERRORS-": self.handle_checkbox_change_event,
            "-SHOW LABELS CHANGES-": self.handle_checkbox_change_event,
            "-SHOW CLASSIFIER PREDICTIONS-": self.handle_checkbox_change_event,
            "-SHOW DATA LABELS-": self.handle_checkbox_change_event,
            "-CIRCLE SELECTING LABELS-": self.handle_circle_selecting_labels_change_event,
            "-UNDO CHANGES-": self.handle_undo_changes_event,
            "-DBM FAST DECODING STRATEGY-": self.handle_decoding_strategy_change_event,
            "-START APPLY CHANGES BTN-": self.handle_start_apply_changes_usage_event,
            "-PAUSE USAGE BTN-": self.handle_pause_apply_changes_usage_event,
        }

        EVENTS[event](event, values)

    def handle_tooltip_checkbox_change_event(self, event, value):
        self.controller.set_show_tooltip_for_dataset_only(not value["-SHOW TOOLTIP OUTSIDE DATASET-"])

    def handle_decoding_strategy_change_event(self, event, values):
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)
        self.window.refresh()

    def _build_plot_(self):
        fig = figure.Figure(figsize=(1, 1))
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        return fig, ax

    def _build_canvas_(self, fig, key, controls_key=None):
        canvas = self.window[key].TKCanvas
        if controls_key is None:
            fig_agg = draw_figure_to_canvas(canvas, fig)
            return canvas, fig_agg
        canvas_controls = self.window[controls_key].TKCanvas
        fig_agg = draw_figure_to_canvas(canvas, fig, canvas_controls)
        return canvas, fig_agg, canvas_controls

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
        img_confidence, color_img = self.controller.img_confidence, self.controller.color_img
        assert(len(img_confidence.shape) == 2)
        assert(color_img.shape[:2] == img_confidence.shape[:2])
        
        mixed_img = np.zeros((img_confidence.shape[0], img_confidence.shape[1], 4))
        mixed_img[:, :, :3] = color_img
        mixed_img[:, :, 3] = img_confidence
        self.axes_image = self.ax.imshow(mixed_img)

        # draw the figure to the canvas
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)
       
    def compute_classifier_metrics(self):
        accuracy, loss, kappa_score = self.controller.compute_classifier_metrics()
        self.window["-CLASSIFIER ACCURACY-"].update(f"Classifier Accuracy: {(100 * accuracy):.2f} %  Loss: {loss:.4f} Kappa score: {kappa_score:.4f}")
        self.update_classifier_performance_canvas()

    def pop_classifier_evaluation(self):
        accuracy, loss, kappa_score = self.controller.pop_classifier_evaluation()
        if accuracy is None and loss is None and kappa_score is None:
            return
        
        self.window["-CLASSIFIER ACCURACY-"].update(f"Classifier Accuracy: {(accuracy):.2f} %  Loss: {loss:.4f} Kappa score: {kappa_score:.4f}")
        self.update_classifier_performance_canvas()

    def handle_compute_inverse_projection_errors_event(self, event, values):
        self.updates_logger.log("Computing inverse projection errors, please wait...")

        self.window['-COMPUTE INVERSE PROJECTION ERRORS-'].hide_row()
        self.window['-PROJECTION ERRORS SECTION-'].update(visible=False)

        self.controller.compute_inverse_projection_errors()
        
        # redraw the figure to the canvas because the layout has changed, otherwise the axes of the figure will not work properly
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)

        self.window['-SHOW INVERSE PROJECTION ERRORS-'].update(visible=True)
        if self.controller.projection_errors is None:
            self.window['-PROJECTION ERRORS SECTION-'].update(visible=True)

        self.updates_logger.log("Inverse projection errors computed!")

    def handle_compute_projection_errors_event(self, event, values):
        self.updates_logger.log("Computing projection errors, please wait...")

        self.window['-COMPUTE PROJECTION ERRORS INTERPOLATION-'].hide_row()
        self.window['-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-'].hide_row()
        self.window['-PROJECTION ERRORS SECTION-'].update(visible=False)
        
        if event == "-COMPUTE PROJECTION ERRORS INTERPOLATION-":
            self.controller.compute_projection_errors(type="interpolated")    
        elif event == "-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-":
            self.controller.compute_projection_errors(type="non_interpolated")    
       
        # redraw the figure to the canvas because the layout has changed, otherwise the axes of the figure will not work properly
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)

        self.window['-SHOW PROJECTION ERRORS-'].update(visible=True)
        if self.controller.inverse_projection_errors is None:
            self.window['-PROJECTION ERRORS SECTION-'].update(visible=True)

        self.updates_logger.log("Finished computing projection errors.")

    def handle_checkbox_change_event(self, event, values):
        img = self.controller.mix_image(values["-SHOW DBM COLOR MAP-"], values["-SHOW DBM CONFIDENCE-"], values["-SHOW INVERSE PROJECTION ERRORS-"], values["-SHOW PROJECTION ERRORS-"])

        if hasattr(self, "axes_image"):
            self.axes_image.remove()

        if hasattr(self, "axes_labels_scatters") and self.axes_labels_scatter is not None:
            self.axes_labels_scatter.remove()
            self.axes_labels_scatter = None

        self.axes_image = self.ax.imshow(img)

        # allow only one of 2 options either show the data labels or the classifier predictions
        show_data_labels, show_classifier_predictions = values["-SHOW DATA LABELS-"], values["-SHOW CLASSIFIER PREDICTIONS-"]
        if show_data_labels:
            values["-SHOW CLASSIFIER PREDICTIONS-"] = False
        if show_classifier_predictions:
            values["-SHOW DATA LABELS-"] = False
            
        if values["-SHOW CLASSIFIER PREDICTIONS-"] or values["-SHOW DATA LABELS-"]:
            encoded_train = self.controller.get_encoded_train_data()
            labels_train = encoded_train[:, 2] if show_classifier_predictions else self.controller.Y_train
            colors_train = [self.colors_mapper[label] for label in labels_train]
            self.axes_labels_scatter = self.ax.scatter(encoded_train[:, 1], encoded_train[:, 0], s=10, c=colors_train)
            
        if hasattr(self, "ax_labels_changes") and self.ax_labels_changes is not None:
            self.ax_labels_changes.remove()
            self.ax_labels_changes = None

        positions_x, positions_y, alphas = self.controller.get_positions_of_labels_changes()
        if values["-SHOW LABELS CHANGES-"] and len(positions_x) > 0 and len(positions_y) > 0 and len(alphas) > 0:
            self.ax_labels_changes = self.ax.scatter(positions_x, positions_y, s=10, c='green', marker='^', alpha=alphas)

        self.fig.canvas.draw_idle()
        self.window["-SHOW DATA LABELS-"].update(values["-SHOW DATA LABELS-"])
        self.window["-SHOW CLASSIFIER PREDICTIONS-"].update(values["-SHOW CLASSIFIER PREDICTIONS-"])
        
    def handle_circle_selecting_labels_change_event(self, event, values):
        self.update_labels_by_circle_select = values["-CIRCLE SELECTING LABELS-"]

    def handle_apply_changes_event(self, event, values):
        self.controller.apply_labels_changes(decoding_strategy=FAST_DBM_STRATEGIES(values["-DBM FAST DECODING STRATEGY-"]), epochs=int(values["-DBM RELABELING CLASSIFIER EPOCHS-"]))
        self.initialize_plots()
        self.compute_classifier_metrics()
        self.handle_checkbox_change_event(event, values)
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)
       
        if self.main_gui is not None and hasattr(self.main_gui, "handle_changes_in_dbm_plotter"):
            self.main_gui.handle_changes_in_dbm_plotter()

    def handle_undo_changes_event(self, event, values):
        self.controller.stop_timer()
        try:
            self.controller.undo_changes()
            self.pop_classifier_evaluation()
            self.controller.user_allowed_interaction_iterations += 1
            self.window["-USAGE ITERATIONS LEFT TEXT-"].update(f"Usage iterations left: {self.controller.user_allowed_interaction_iterations}")
            self.updates_logger.log("Undone changes successfully")
        except Exception as e:
            self.updates_logger.error("Failed to undo changes: " + str(e))
            
        self.initialize_plots()
        self.handle_checkbox_change_event(event, values)
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)
        # start a timer
        self.controller.start_timer(self.window["-APPLY CHANGES TIMER TEXT-"])

    def handle_show_classifier_performance_history_event(self, event=None, values=None):
        try:
            times, accuracies, losses, kappas = self.controller.get_classifier_performance_history()
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))

            for ax in [ax1, ax2, ax3]:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)

            ax1.set_title("Classifier accuracy history")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Accuracy (%)")

            ax2.set_title("Kappa score history")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Kappa score")

            ax3.set_title("Classifier loss history")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Loss")


            ax1.plot(times, accuracies, marker="o")
            ax2.plot(times, kappas, marker="o")
            ax3.plot(times, losses, marker="o")

            plt.show()
        except Exception as e:
            self.updates_logger.error("Failed to show classifier performance history: " + str(e))

    def handle_start_apply_changes_usage_event(self, event, values):
        # hide the start usage btn
        self.window["-START APPLY CHANGES SECTION-"].update(visible=False)
        
        # allow user to interact with the dbm plot by clicking
        self.initialize_plots()
        self.handle_checkbox_change_event(event, values)
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)
        
        # start a timer
        self.controller.start_timer(self.window["-APPLY CHANGES TIMER TEXT-"])
        # reveal the apply changes section to the ui
        self.window["-APPLY CHANGES SECTION-"].update(visible=True)   
        self.window["-PROJECTION ERRORS SECTION-"].update(visible=False)
        
    def handle_pause_apply_changes_usage_event(self, event, values):
        # stop timer if any
        self.controller.stop_timer()
        # hide the apply changes section
        self.window["-APPLY CHANGES SECTION-"].update(visible=False)
        # reveal the start apply changes usage section
        self.window["-START APPLY CHANGES SECTION-"].update(visible=True)
        # reveal the compute errors buttons section
        errors_computed = self.controller.projection_errors is not None and self.controller.inverse_projection_errors is not None
        self.window["-PROJECTION ERRORS SECTION-"].update(visible=not errors_computed)

        # disable user to interact with the dbm plot by clicking
        self.initialize_plots(connect_click_event=False)
        self.handle_checkbox_change_event(event, values)
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)

    def update_classifier_performance_canvas(self):
        times, accuracies, _, _ = self.controller.get_classifier_performance_history()
        self.classifier_performance_fig, self.classifier_performance_ax = self._build_plot_()
        self.classifier_performance_ax.set_axis_on()
        self.classifier_performance_ax.set_title("Classifier performance history")
        #self.classifier_performance_ax.set_xlabel("Time")
        self.classifier_performance_ax.set_ylabel("Accuracy (%)")
        self.classifier_performance_fig.canvas.mpl_connect('button_press_event', self.handle_show_classifier_performance_history_event)

        self.classifier_performance_ax.plot(times, accuracies, marker="o")

        self.classifier_performance_fig_agg = draw_figure_to_canvas(self.classifier_performance_canvas, self.classifier_performance_fig)
        self.window.refresh()
