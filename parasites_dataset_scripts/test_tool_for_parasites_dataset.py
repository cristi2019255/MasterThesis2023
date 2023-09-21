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

from math import sqrt
import tensorflow as tf
import os
import numpy as np
import random
from src.decision_boundary_mapper.DBM.AbstractDBM import FAST_DBM_STRATEGIES
from src.decision_boundary_mapper.DBM.DBM import DBM
from src.decision_boundary_mapper.utils.dataReader import import_folder_dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score

SEED = 42
DATA_FOLDER = './data/parasites_focus_plane_divided/ovos/resized_64_grayscale'
DATASET_NAME = "_".join(DATA_FOLDER.split(os.sep)[-2:])

# make the decision boundary map pretty, by adding the colors and the confidence
COLORS_MAPPER = {
        -2: [0, 0, 0],  # setting original test data to black
        -1: [1, 1, 1],  # setting original train data to white
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

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def import_data():
    # import the dataset
    (X_train, Y_train), (X_test, Y_test) = import_folder_dataset(DATA_FOLDER)

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return X_train, X_test, Y_train, Y_test

def import_2d_data(projection='t-SNE'):
    # upload the 2D projection of the data
    with open(os.path.join("tmp", DATASET_NAME, "DBM", projection, "train_2d.npy"), "rb") as f:
        X2d_train = np.load(f)
    with open(os.path.join("tmp", DATASET_NAME, "DBM", projection, "test_2d.npy"), "rb") as f:
        X2d_test = np.load(f)
    return X2d_train, X2d_test

def import_classifier():
    # upload a classifier
    # !!! the classifier must be a tf.keras.models.Model !!!
    # !!! change the next line with the path to your classifier !!!
    classifier_path = os.path.join("tmp", DATASET_NAME, "classifier")
    classifier = tf.keras.models.load_model(classifier_path)
    return classifier

def import_nn(nn_path):
    nn = tf.keras.models.load_model(nn_path)
    return nn

def train_nn(data2d_path, save_path, Xnd):
    output_shape = (64, 64)
    output_size = 1
    for i in output_shape:
        output_size *= i
    print("Output shape ", output_shape, " size ", output_size)
    
    decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform',
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform',
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(output_size, activation='sigmoid', kernel_initializer='he_uniform'),
            tf.keras.layers.Reshape(output_shape)
        ], name="decoder")
    
    input_layer = tf.keras.Input(shape=(2,), name="input")

    neural_network = tf.keras.models.Model(inputs=input_layer,
                                            outputs=[decoder(input_layer)],
                                            name="NNInv")

    neural_network.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss="mse",
                            metrics=["accuracy"])

    neural_network.summary()
    
    with open(data2d_path, 'rb') as f:
        X2d = np.load(f)
    
    neural_network.fit(X2d, Xnd,
                       epochs=10,
                       batch_size=128,
                       shuffle=True,
                       validation_split=0.2,
                    )
    
    neural_network.save(save_path, save_format='tf')
    
    return neural_network

def show_predictions(neural_network, classifier, dataNd: np.ndarray, labels: np.ndarray, data2d: np.ndarray, n_samples: int = 16):
    
    #dataNd = dataNd[:n_samples+1]
    #data2d = data2d[:n_samples+1]
    decoded = neural_network.predict(data2d)
    predicted = classifier.predict(decoded)
    predicted_labels = np.argmax(predicted, axis=1)

    print("Accuracy: ", accuracy_score(labels, predicted_labels))
    
    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        print(f"Class {i}:  predicted {np.count_nonzero(predicted_labels == i)}, real {np.count_nonzero(labels == i)}")

    plt.figure(figsize=(20, 10))
    m = int(sqrt(n_samples))
    n = m * m

    print("X2d shape: ", data2d.shape)
    print("Xnd shape: ", dataNd.shape)
    print("Y shape: ", labels.shape)
    

    for i in range(1, 2*n, 2):
        plt.subplot(m, 2*m, i)
        plt.imshow(dataNd[i], cmap='gray')
        plt.title(f"Actual: {labels[i]}", color='green' if predicted_labels[i]== labels[i] else 'red', fontsize=12)
        plt.axis('off')
        plt.subplot(m, 2*m, i+1)
        plt.imshow(decoded[i], cmap='gray')
        plt.title(f"Predicted: {predicted_labels[i]}", color='green' if predicted_labels[i] == labels[i] else 'red', fontsize=12)
        plt.axis('off')

    plt.show()

def view_2d_projection(data2d_path, Y):

    with open(data2d_path, "rb") as f:
        X2d = np.load(f)
    print("X2d shape: ", X2d.shape)
    print("Y shape: ", Y.shape)
    plot_2d(X2d, Y)


def view_classifier_predictions_2d(data2d_path, nn_path):
    with open(data2d_path, "rb") as f:
        X2d = np.load(f)
    
    # import the classifier
    classifier = import_classifier()
    nninv = import_nn(nn_path)
    
    Xnd = nninv.predict(X2d)
    Y_pred = classifier.predict(Xnd)
    Y = np.array([np.argmax(y) for y in Y_pred])
    print(Y.shape)
    
    with open("classifier_predictions.npy", "wb") as f:
        np.save(f, Y)
    
    plot_2d(X2d, Y)

def plot_2d(X, Y):
    color_map = [COLORS_MAPPER[y] for y in Y]
    x, y = X[:, 0], X[:, 1]
    plt.scatter(x, y, c=color_map)
    
    patches = []
    for value in np.unique(Y):
        color = COLORS_MAPPER[value]
        if value == -1:
            label = "Original train data"
        elif value == -2:
            label = "Original test data"
        else:
            label = f"Value region: {value}"

        patches.append(Patch(color=color, label=label))

    plt.legend(handles=patches)
    plt.show()
    
def plot_dbm(img, labels, X2d):
    resolution = len(img)
    X2d = (1 - X2d) * (resolution - 1)
    X2d = X2d.astype(int)
    img = np.rot90(img)
    for [j, i] in X2d:
        img[i, j] = -1
    
    patches = []
    for value in np.unique(labels):
        color = COLORS_MAPPER[value]
        if value == -1:
            label = "Original train data"
        elif value == -2:
            label = "Original test data"
        else:
            label = f"Value region: {value}"

        patches.append(Patch(color=color, label=label))

    color_img = np.zeros((resolution, resolution, 3))
    for i in range(resolution):
        for j in range(resolution):
            color_img[i,j] = COLORS_MAPPER[img[i,j]]

    plt.legend(handles=patches)
    plt.imshow(color_img)
    plt.show()
    
def generate_map(neural_network, classifier, resolution = 256):
    points2d = np.array([(i/resolution, j/resolution) for i in range(0, resolution) for j in range(0, resolution)])
    xnd = neural_network.predict(points2d)
    predictions = classifier.predict(xnd)
    predicted_labels = np.argmax(predictions, axis=1)
    print(predicted_labels.shape)
    img = predicted_labels.reshape((resolution, resolution))
    return img, predicted_labels

def test():
# import the dataset
    X_train, X_test, Y_train, Y_test = import_data()

    # import the classifier
    classifier = import_classifier()

    # create the DBM
    dbm = DBM(classifier=classifier)

    X2d_train, X2d_test = import_2d_data()

    # use the DBM to get the decision boundary map, if you don't have the 2D projection of the data
    # the DBM will get it for you, you just need to specify the projection method you would like to use (t-SNE, PCA or UMAP)
    img, img_confidence, _, _ = dbm.generate_boundary_map(X_train,
                                                          X_test,
                                                          X2d_train=X2d_train,
                                                          X2d_test=X2d_test,
                                                          resolution=256,
                                                          load_folder=os.path.join("tmp", DATASET_NAME, "DBM"),
                                                          fast_decoding_strategy=FAST_DBM_STRATEGIES.NONE,
                                                          )


    color_img = np.zeros((img.shape[0], img.shape[1], 4))
    for i, j in np.ndindex(img.shape):
        color_img[i, j] = COLORS_MAPPER[img[i, j]] + [img_confidence[i, j]]

    # plot the decision boundary map
    plt.title("Decision boundary map")
    plt.axis("off")
    plt.imshow(color_img)
    plt.show()


# Call the above function with seed value
set_global_determinism(seed=SEED)

data2d_path = os.path.join("tmp", DATASET_NAME, "DBM", "t-SNE", "train_2d.npy")
nn_path = os.path.join("tmp", DATASET_NAME, "DBM", "t-SNE", "NNInv")

(X_train, Y_train), (X_test, Y_test) = import_folder_dataset(DATA_FOLDER)
classifier = import_classifier()

nn = train_nn(data2d_path, save_path="NNinv_test", Xnd=X_train)
nn = import_nn("NNinv_test")

with open(data2d_path, "rb") as f:
    X2d = np.load(f)

show_predictions(nn, classifier, X_train, Y_train, X2d, n_samples=9)

"""
img, labels = generate_map(nn, classifier)
with open("dbm_test.npy", "wb") as f:
    np.save(f, img)
with open("dbm_test_labels.npy", "wb") as f:
    np.save(f, labels)
with open("dbm_test.npy", "rb") as f:
    img = np.load(f)
with open("dbm_test_labels.npy", "rb") as f:
    labels = np.load(f)


plot_dbm(img, labels, X2d)
"""

#view_2d_projection(data2d_path, Y_train)

#view_classifier_predictions_2d(data2d_path, nn_path)

#test()


