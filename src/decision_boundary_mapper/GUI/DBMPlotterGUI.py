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

from .. import Logger, LoggerGUI, FAST_DBM_STRATEGIES
from .DBMPlotterController import DBMPlotterController
from .DBMPlotterController import EPOCHS_FOR_REFIT, EPOCHS_FOR_REFIT_RANGE
from ..utils import TRAIN_DATA_POINT_MARKER, TEST_DATA_POINT_MARKER, BLACK_COLOR, WHITE_COLOR, RED_COLOR, GREEN_COLOR, YELLOW_COLOR, RIGHTS_MESSAGE_1, RIGHTS_MESSAGE_2, APP_PRIMARY_COLOR, APP_FONT

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


def generate_color_mapper():
    colors_mapper = {
        # setting original test data to black
        TEST_DATA_POINT_MARKER: [0, 0, 0],
        # setting original train data to white
        TRAIN_DATA_POINT_MARKER: [1, 1, 1],
        0: [1, 0, 0],
        1: [0, 1, 0],
        2: [0, 0, 1],
        3: [1, 1, 0],
        4: [0, 1, 1],
        5: [1, 0, 1],
        6: [0.5, 0.5, 0.5],
        7: [0.5, 0, 0],
        8: [0, 0.5, 0],
        9: [0, 0, 0.5]
    }
    # setting the rest of the colors
    for i in range(10, 100):
        colors_mapper[i] = [i/200, i/200, i/200]
    return colors_mapper


# Generating initial settings
COLORS_MAPPER = generate_color_mapper()

TITLE = "Decision Boundary Map & Errors"
WINDOW_SIZE = (1650, 1000)
INFORMATION_CONTROLS_MESSAGE = "To change label(s) of a data point click on the data point,\n or select the data point by including them into a circle.\nPress any digit key to indicate the new label.\nPress 'Enter' to confirm the new label. Press 'Esc' to cancel the action.\nTo remove a change just click on the data point.\nPress 'Apply Changes' to update the model."
DBM_WINDOW_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "dbm_plotter_icon.png")


class DBMPlotterGUI:
    
    def __init__(self,
                 dbm_model,
                 img, img_confidence,
                 X_train, Y_train,
                 X_test, Y_test,
                 encoded_train, encoded_test,
                 save_folder,
                 X_train_2d = None, X_test_2d = None,
                 projection_technique=None,
                 logger=None,
                 main_gui=None,
                 class_name_mapper= lambda x: str(x)
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
            class_name_mapper (function, optional): The function which is meant for mapping class names to their corresponding values. Defaults to lambda x -> str(x).
        """
        
        self.main_gui = main_gui  # reference to main window
        self.class_name_mapper = class_name_mapper
        if logger is None:
            self.console = Logger(name="DBMPlotterGUI")
        else:
            self.console = logger
        
        self.controller = DBMPlotterController(logger,
                                                dbm_model,
                                                img, img_confidence,
                                                X_train, Y_train,
                                                X_test, Y_test,
                                                X_train_2d, X_test_2d,
                                                encoded_train, encoded_test,
                                                save_folder,
                                                projection_technique)
        self.initialize()
        

    def initialize(self):
        self.color_img, legend = self.controller.build_2D_image(colors_mapper=COLORS_MAPPER, class_name_mapper=self.class_name_mapper)
        # --------------------- Plotter related ---------------------
        self.classifier_performance_fig, self.classifier_performance_ax = self._build_plot_()
        self.fig, self.ax = self._build_plot_()
        self.controller.build_annotation_mapper(self.fig, self.ax)
        self.fig.legend(handles=legend, borderaxespad=0.)

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
                            key="-DBM FAST DECODING STRATEGY-",
                            background_color=WHITE_COLOR,
                            text_color=BLACK_COLOR,
                            button_background_color=APP_PRIMARY_COLOR,
                            readonly=True,
                        ),
                    ],
                    [   
                        sg.Text(f"Number of epochs:", font=APP_FONT, key="-DBM RELABELING CLASSIFIER EPOCHS TEXT-"),
                        sg.Slider(range=EPOCHS_FOR_REFIT_RANGE, 
                               default_value=EPOCHS_FOR_REFIT, 
                               expand_x=True, 
                               enable_events=False,
                               orientation='horizontal',
                               trough_color=WHITE_COLOR,
                               font=APP_FONT,
                               #button_color=(WHITE_COLOR, APP_PRIMARY_COLOR),
                               tooltip="Select the number of epochs for which \nthe classifier will be retrained.",
                               key="-DBM RELABELING CLASSIFIER EPOCHS-")
                    ],
                    [
                        sg.Button("Apply Changes", font=APP_FONT, expand_x=True, key="-APPLY CHANGES-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR)),
                        sg.Button("Undo Changes", font=APP_FONT, expand_x=True, key="-UNDO CHANGES-", button_color=(WHITE_COLOR, APP_PRIMARY_COLOR)),
                    ],
                    [
                        sg.HSeparator()
                    ],
                    buttons_proj_errs[0],
                    buttons_proj_errs[1],
                    buttons_inv_proj_errs,
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
                           icon=DBM_WINDOW_ICON_PATH,
                           element_justification='center',
                           )

        window.finalize()
        window.maximize()
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
            "-SHOW DBM COLOR MAP-": self.handle_checkbox_change_event,
            "-SHOW DBM CONFIDENCE-": self.handle_checkbox_change_event,
            "-SHOW INVERSE PROJECTION ERRORS-": self.handle_checkbox_change_event,
            "-SHOW PROJECTION ERRORS-": self.handle_checkbox_change_event,
            "-SHOW LABELS CHANGES-": self.handle_checkbox_change_event,
            "-SHOW CLASSIFIER PREDICTIONS-": self.handle_checkbox_change_event,
            "-CIRCLE SELECTING LABELS-": self.handle_circle_selecting_labels_change_event,
            "-UNDO CHANGES-": self.handle_undo_changes_event,
            "-DBM FAST DECODING STRATEGY-": self.handle_decoding_strategy_change_event,
        }

        EVENTS[event](event, values)

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
       
    def compute_classifier_metrics(self, epochs=None):
        accuracy, loss = self.controller.compute_classifier_metrics(epochs)
        self.window["-CLASSIFIER ACCURACY-"].update(f"Classifier Accuracy: {(100 * accuracy):.2f} %  Loss: {loss:.2f}")
        self.update_classifier_performance_canvas()

    def pop_classifier_evaluation(self):
        accuracy, loss = self.controller.pop_classifier_evaluation()
        if accuracy is None or loss is None:
            return
        
        self.window["-CLASSIFIER ACCURACY-"].update(f"Classifier Accuracy: {(accuracy):.2f} %  Loss: {loss:.2f}")
        self.update_classifier_performance_canvas()

    def handle_compute_inverse_projection_errors_event(self, event, values):
        self.window['-COMPUTE INVERSE PROJECTION ERRORS-'].hide_row()
        self.updates_logger.log("Computing inverse projection errors, please wait...")
        self.controller.compute_inverse_projection_errors()
        self.updates_logger.log("Inverse projection errors computed!")
        self.window['-SHOW INVERSE PROJECTION ERRORS-'].update(visible=True)
        
        # redraw the figure to the canvas because the layout has changed, otherwise the axes of the figure will not work properly
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)

    def _set_loading_proj_errs_state_(self):
        self.window['-COMPUTE PROJECTION ERRORS INTERPOLATION-'].hide_row()
        self.window['-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-'].hide_row()
        self.updates_logger.log("Computing projection errors, please wait...")

    def handle_compute_projection_errors_event(self, event, values):
        self._set_loading_proj_errs_state_()
        if event == "-COMPUTE PROJECTION ERRORS INTERPOLATION-":
            self.controller.compute_projection_errors(type="interpolated")    
        elif event == "-COMPUTE PROJECTION ERRORS INVERSE PROJECTION-":
            self.controller.compute_projection_errors(type="non_interpolated")    
    
        self.updates_logger.log("Finished computing projection errors.")
        self.window['-SHOW PROJECTION ERRORS-'].update(visible=True)
        
        # redraw the figure to the canvas because the layout has changed, otherwise the axes of the figure will not work properly
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)

    def handle_checkbox_change_event(self, event, values):
        img = self.controller.mix_image(values["-SHOW DBM COLOR MAP-"], values["-SHOW DBM CONFIDENCE-"], values["-SHOW INVERSE PROJECTION ERRORS-"], values["-SHOW PROJECTION ERRORS-"])

        if hasattr(self, "axes_image"):
            self.axes_image.remove()

        if hasattr(self, "axes_classifier_scatter") and self.axes_classifier_scatter is not None:
            self.axes_classifier_scatter.set_visible(False)
            self.axes_classifier_scatter = None

        self.axes_image = self.ax.imshow(img)

        if values["-SHOW CLASSIFIER PREDICTIONS-"]:
            encoded_train = self.controller.get_encoded_train_data()
            colors = [COLORS_MAPPER[label] for label in encoded_train[:, 2]]
            self.axes_classifier_scatter = self.ax.scatter(encoded_train[:, 1], encoded_train[:, 0], s=10, c=colors)

        if hasattr(self, "ax_labels_changes") and self.ax_labels_changes is not None:
            self.ax_labels_changes.set_visible(False)
            self.ax_labels_changes = None

        positions_x, positions_y, alphas = self.controller.get_positions_of_labels_changes()
        if values["-SHOW LABELS CHANGES-"] and len(positions_x) > 0 and len(positions_y) > 0 and len(alphas) > 0:
            self.ax_labels_changes = self.ax.scatter(positions_x, positions_y, s=10, c='green', marker='^', alpha=alphas)

        self.fig.canvas.draw_idle()
        

    def handle_circle_selecting_labels_change_event(self, event, values):
        self.update_labels_by_circle_select = values["-CIRCLE SELECTING LABELS-"]

    def handle_apply_changes_event(self, event, values):
        self.controller.apply_labels_changes(decoding_strategy=FAST_DBM_STRATEGIES(values["-DBM FAST DECODING STRATEGY-"]), epochs=int(values["-DBM RELABELING CLASSIFIER EPOCHS-"]))
        self.initialize()
        self.compute_classifier_metrics(int(values["-DBM RELABELING CLASSIFIER EPOCHS-"]))
        self.handle_checkbox_change_event(event, values)
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)
       

        self.updates_logger.log("Changes applied successfully!")

        if self.main_gui is not None and hasattr(self.main_gui, "handle_changes_in_dbm_plotter"):
            self.main_gui.handle_changes_in_dbm_plotter()

    def handle_undo_changes_event(self, event, values):
        try:
            self.controller.undo_changes()
            self.pop_classifier_evaluation()

            self.updates_logger.log("Undone changes successfully")
        except Exception as e:
            self.updates_logger.error("Failed to undo changes: " + str(e))
            
        self.initialize()
        self.handle_checkbox_change_event(event, values)
        self.fig_agg = draw_figure_to_canvas(self.canvas, self.fig, self.canvas_controls)
       

    def handle_show_classifier_performance_history_event(self, event=None, values=None):
        try:
            times, accuracies, losses = self.controller.get_classifier_performance_history()
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

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

            ax1.plot(times, accuracies, marker="o")
            ax2.plot(times, losses, marker="o")
            plt.show()
        except Exception as e:
            self.updates_logger.error("Failed to show classifier performance history: " + str(e))

    def update_classifier_performance_canvas(self):
        times, accuracies, _ = self.controller.get_classifier_performance_history()
        self.classifier_performance_fig, self.classifier_performance_ax = self._build_plot_()
        self.classifier_performance_ax.set_axis_on()
        self.classifier_performance_ax.set_title("Classifier performance history")
        #self.classifier_performance_ax.set_xlabel("Time")
        self.classifier_performance_ax.set_ylabel("Accuracy (%)")
        self.classifier_performance_fig.canvas.mpl_connect('button_press_event', self.handle_show_classifier_performance_history_event)

        self.classifier_performance_ax.plot(times, accuracies, marker="o")

        self.classifier_performance_fig_agg = draw_figure_to_canvas(self.classifier_performance_canvas, self.classifier_performance_fig)
        self.window.refresh()
