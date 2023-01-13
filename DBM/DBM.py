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

from DBM.DBMInterface import DBMInterface
from DBM.DBMInterface import DBM_DEFAULT_RESOLUTION
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

class DBM(DBMInterface):
    
    def __init__(self, classifier, logger=None):
        super().__init__(classifier, logger)
    
    def build_decoder(self, output_shape):
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='linear'),
            tf.keras.layers.Reshape(output_shape)
        ], name="decoder")

        self.decoder_classifier = tf.keras.Sequential([
            self.decoder,
            self.classifier
        ], name="decoder_classifier")  
        
        input_layer = tf.keras.Input(shape=(None, 2), name="input")
                                             
        decoder_classifier_output = self.decoder_classifier(input_layer)
        decoder_output = self.decoder(input_layer)
        
            
        self.invNN = tf.keras.models.Model(inputs=input_layer, 
                                            outputs=[decoder_output, 
                                                    decoder_classifier_output], 
                                            name="invNN")
        
        optimizer = tf.keras.optimizers.Adam()
        
        self.invNN.compile(optimizer=optimizer, 
                                loss={"decoder":"binary_crossentropy",
                                      "decoder_classifier": "sparse_categorical_crossentropy"}, 
                                metrics=['accuracy'])
                                          
            
    def fit(self, 
            X2d_train, X_train, Y_train, 
            X2d_test, X_test, Y_test, 
            epochs=10, batch_size=128, load_folder=None):
        """ Trains the classifier on the given data set.

        Args:
            X_train (np.array): Training data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Y_train (np.array): Training data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            X_test (np.array): Testing data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Y_test (np.array): Testing data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            batch_size (int, optional): Train batch size. Defaults to 128.
        """
        self.build_decoder(output_shape=X_train.shape[1:])  
        self.invNN.fit(X2d_train, 
                       [X_train, Y_train], 
                       epochs=epochs, 
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(X2d_test, [X_test, Y_test])
                       )
        
    def generate_boundary_map(self, 
                              X_train, 
                              Y_train, 
                              X_test, 
                              Y_test,
                              train_epochs=10, 
                              train_batch_size=128,
                              resolution=DBM_DEFAULT_RESOLUTION,
                              encoder='t-SNE'
                              ):
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
        if encoder == 't-SNE':
            encoder = TSNE(n_components=2, random_state=0)
            self.console.log("Transforming the data to 2D using t-SNE")
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            X2d_train = encoder.fit_transform(X_train_flat)
            X2d_test = encoder.fit_transform(X_test_flat)
            self.console.log("Finished transforming the data to 2D using t-SNE")
            self.console.log("Starting training the DBM")
            self.fit(X2d_train, X_train, Y_train, X2d_test, X_test, Y_test, epochs=train_epochs, batch_size=train_batch_size)
            self.console.log("Finishing training the DBM")
            self.invNN.save("invNN")
            
            min_x = min(np.min(X2d_train[:,0]), np.min(X2d_test[:,0]))
            max_x = max(np.max(X2d_train[:,0]), np.max(X2d_test[:,0]))
            min_y = min(np.min(X2d_train[:,1]), np.min(X2d_test[:,1]))
            max_y = max(np.max(X2d_train[:,1]), np.max(X2d_test[:,1]))
            
            space2d = np.array([(i / resolution * (max_x - min_x) + min_x, j / resolution * (max_y - min_y) + min_y) for i in range(resolution) for j in range(resolution)])
        
            self.console.log("Decoding the 2D space... 2D -> nD -> 1D")
            predictions = self.decoder_classifier.predict(space2d)
            predicted_labels = np.array([np.argmax(p) for p in predictions])
            img = predicted_labels.reshape((resolution, resolution))
        
            print(img)

            # TODO: generate the training and testing corpus embeddings in the 2D space of the image
            # TODO: refactoring as written in hurry
            # TODO: delegate the NN part to another class and use DBM as a wrapper for the NN
            
            plt.imshow(img, cmap='gray')
            plt.show()
            
        return img, None, None