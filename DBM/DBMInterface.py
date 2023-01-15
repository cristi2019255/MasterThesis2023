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

from utils.Logger import Logger


DBM_DEFAULT_RESOLUTION = 256


class DBMInterface:
    
    def __init__(self, classifier, logger=None):
        if logger is None:
            self.console = Logger(name="Decision Boundary Mapper - DBM, using Autoencoder")
        else:
            self.console = logger
        
        self.console.log("Loaded classifier: " + classifier.name + "")
        self.classifier = classifier
        
    def fit(self, X_train, Y_train, X_test, Y_test, epochs=10, batch_size=128, load_folder = None):
        """ Trains the classifier on the given data set.

        Args:
            X_train (np.array): Training data set
            Y_train (np.array): Training data labels
            X_test (np.array): Testing data set
            Y_test (np.array): Testing data labels
            epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            batch_size (int, optional): Train batch size. Defaults to 128.
        """
        pass
    
    def generate_boundary_map(self, 
                              X_train, Y_train, X_test, Y_test,
                              train_epochs=10, 
                              train_batch_size=128,
                              resolution=DBM_DEFAULT_RESOLUTION):
        """ Generates a 2D boundary map of the classifier's decision boundary.

        Args:
            X_train (np.array): Training data set
            Y_train (np.array): Training data labels
            X_test (np.array): Testing data set
            Y_test (np.array): Testing data labels
            train_epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            train_batch_size (int, optional): Train batch size. Defaults to 128.
            show_predictions (bool, optional): If set to true 10 prediction examples are shown. Defaults to True.
            resolution (_type_, optional): _description_. Defaults to DBM_DEFAULT_RESOLUTION.
        
        Returns:
            np.array: A 2D numpy array with the decision boundary map
        
        """
        pass