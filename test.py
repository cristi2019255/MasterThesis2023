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

from scipy import interpolate
import time
from src.decision_boundary_mapper import DBM, FAST_DBM_STRATEGIES
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.decision_boundary_mapper.DBM import SDBM
from src.decision_boundary_mapper.DBM.SDBM.SDBM import NNArchitecture
from src.decision_boundary_mapper.GUI import DBMPlotterGUI
from src.decision_boundary_mapper.utils.dataReader import import_dataset, import_mnist_dataset, import_folder_dataset
from experiments.scripts import import_data

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

FAST_DECODING_STRATEGY = FAST_DBM_STRATEGIES.CONFIDENCE_INTERPOLATION


def import_classifier():
    # upload a classifier
    # !!! the classifier must be a tf.keras.models.Model !!!
    # !!! change the next line with the path to your classifier !!!
    classifier_path = os.path.join("tmp", "MNIST", "classifier")
    classifier = tf.keras.models.load_model(classifier_path)
    return classifier


def import_2d_data():
    # upload the 2D projection of the data
    with open(os.path.join("tmp", "MNIST", "DBM", "t-SNE", "train_2d.npy"), "rb") as f:
        X2d_train = np.load(f)
    with open(os.path.join("tmp", "MNIST", "DBM", "t-SNE", "test_2d.npy"), "rb") as f:
        X2d_test = np.load(f)
    return X2d_train, X2d_test

def test():
    
    X_train, X_test, _, _ = import_data()
    classifier = import_classifier()
    X2d_train, X2d_test = import_2d_data()
    dbm = DBM(classifier)
    resolution = 256

    dbm.generate_boundary_map(X_train,
                              X_test,
                              X2d_train,
                              X2d_test,
                              resolution=512,
                              fast_decoding_strategy=FAST_DBM_STRATEGIES.NONE,
                              load_folder=os.path.join("tmp", "MNIST", "DBM"),
                              projection='t-SNE')

    if FAST_DECODING_STRATEGY == FAST_DBM_STRATEGIES.BINARY:
        img_path = "img_B.npy"
        img_confidence_path = "img_confidence_B.npy"
        start = time.time()
        img1, img_confidence1, _ = dbm._get_img_dbm_fast_(resolution)
        end = time.time()
        print("Fast decoding time: ", end - start)
    elif FAST_DECODING_STRATEGY == FAST_DBM_STRATEGIES.CONFIDENCE_BASED:
        img_path = "img_C.npy"
        img_confidence_path = "img_confidence_C.npy"

        start = time.time()
        img1, img_confidence1, _ = dbm._get_img_dbm_fast_confidences_strategy(resolution)
        end = time.time()
        print("Fast decoding time: ", end - start)
    else:
        
        img_path = "img_F.npy"
        img_confidence_path = "img_confidence_F.npy"
        """
        start = time.time()
        img1, img_confidence1, _ = dbm._get_img_dbm_fast_confidence_interpolation_strategy(resolution, interpolation_method="cubic")
        end = time.time()
        print("Fast decoding time: ", end - start)
        """
        start = time.time()
        img1, img_confidence1, _ = dbm._get_img_dbm_fast_confidence_interpolation_strategy(resolution, interpolation_method="linear", initial_resolution=32)
        end = time.time()
        print("Fast decoding time: ", end - start)

    with open(img_path, "wb") as f:
        np.save(f, img1)
    with open(img_confidence_path, "wb") as f:
        np.save(f, img_confidence1)

def test3():
    X_train, X_test, Y_train, Y_test = import_data()
    classifier = import_classifier()
    X2d_train, X2d_test = import_2d_data()
    dbm = DBM(classifier)
    resolution = 256

    dbm.generate_boundary_map(X_train, 
                              X_test,
                              X2d_train,
                              X2d_test,
                              resolution=10,
                              fast_decoding_strategy=FAST_DBM_STRATEGIES.BINARY,
                              load_folder=os.path.join("tmp", "MNIST", "DBM"),
                              projection='t-SNE')

    start = time.time()
    errors = dbm.generate_inverse_projection_errors(resolution=500)
    with open("inverse_projection_errors_big.npy", "wb") as f:
        np.save(f, errors)
    end = time.time()
    print("Time: ", end - start)


def show_errors():      
    if FAST_DECODING_STRATEGY == FAST_DBM_STRATEGIES.BINARY:
        img_path = "img_B.npy"
    elif FAST_DECODING_STRATEGY == FAST_DBM_STRATEGIES.CONFIDENCE_BASED:
        img_path = "img_C.npy"
    else:
        img_path = "img_F.npy"
        
    TEST_FILE_PATH = f"/Users/cristiangrosu/Desktop/code_repo/MasterThesis2023/{img_path}"
    GROUND_TRUTH_FILE_PATH = "/Users/cristiangrosu/Desktop/code_repo/MasterThesis2023/tmp/MNIST/DBM/t-SNE/boundary_map.npy"
    
    with open(TEST_FILE_PATH, "rb") as f:
        errors = np.load(f)

    with open(GROUND_TRUTH_FILE_PATH, "rb") as f:
        errors2 = np.load(f)

    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(errors, cmap='gray')
    ax1.axis('off')
    #ax2.imshow(errors2)
    
    errors_count = 0
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            if errors[i, j] != errors2[i, j]:
                ax1.plot(j, i, 'ro')
                errors_count += 1                

    print("Errors count: ", errors_count)
    print("Errors percentage:", errors_count / (errors.shape[0] * errors.shape[1]) * 100, "%")

    plt.show()


def test_interpolation():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    resolution = 10
    sparse_map = [(0, 0, 0), (2, 1, 1), (0, 3, 2), (4, 1, 3), (4, 4, 4), (0, 4, 5),
                  (4, 0, 6), (2, 2, 7), (2, 3, 8), (3, 2, 9), (3, 3, 10), (8, 8, 10)]
    X, Y, Z = [], [], []
    for (x, y, z) in sparse_map:
        X.append(x)
        Y.append(y)
        Z.append(z)
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    xi = np.linspace(0, resolution-1, resolution)
    yi = np.linspace(0, resolution-1, resolution)
    grid = interpolate.griddata((X, Y), Z, (xi[None, :], yi[:, None]), method="cubic")

    #print(grid)
    xx, yy = np.meshgrid(xi, yi)
    #iterpolator = interpolate.SmoothBivariateSpline(X, Y, Z, kx=2, ky=2)
    iterpolator = interpolate.Rbf(X, Y, Z, epsilon=2, function='multiquadric')
    z = iterpolator(xx, yy)

    #print(z)

    #grid = np.array(grid)
    z = np.array(z)

    #grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
    z = (z - np.min(z)) / (np.max(z) - np.min(z))

    ax1.imshow(grid)
    ax2.imshow(z)
    plt.show()


def show_img(path):
    with open(path, "rb") as f:
        img = np.load(f)
    plt.imshow(img)
    plt.show()

def show_grid():
    fig, ax = plt.subplots(1,1)
    x = np.linspace(0, 1, 4, endpoint=False)
    y = np.linspace(0, 1, 4, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    
    ax.axis('off')
    ax.scatter(xx, yy)
    ax.scatter([0.5, 0.25, 0.25, 0.5], [0.5, 0.5, 0.25, 0.25], color='red')
    ax.fill_between([0.5, 0.25], 0.5, 0.25, color='red', alpha=0.3)
    ax.plot([0.25, 0.5], [0.5,0.5], 'r--')
    ax.plot([0.25, 0.5], [0.25,0.25], 'r--')
    ax.plot([0.5, 0.5], [0.25,0.5], 'r--')
    ax.plot([0.25, 0.25], [0.25,0.5], 'r--')
    ax.scatter([0.375], [0.375], color='black')
    ax.plot()
    
    plt.show()
    
def show_fig_3_10():
    annotation_offset_x, annotation_offset_y = 0.05, 0.05
    fontsize = 16
 
    fig, ax = plt.subplots(1,1, figsize=(15,10))
    #ax.axis("off")
    ax.set_xlim((-1.8,1.8))
    ax.set_ylim((-1.6,1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    
    
    central_point = (0,0)
    neighbour_1 = (-1.1, 0)
    neighbour_2 = (0, -1.1)
    neighbour_3 = (0, 1.1)
    neighbour_4 = (1.1, 0)
    
    ax.scatter([neighbour_1[0]],[neighbour_1[1]], color='magenta')
    ax.annotate('label: 1\nconfidence:  0.87', xy=neighbour_1, 
                xytext=(neighbour_1[0] - 12 * annotation_offset_x, neighbour_1[1] + 2 * annotation_offset_y), 
                arrowprops=dict(fc='w', arrowstyle="-|>"), fontsize=fontsize,fontweight="bold")
    
    ax.scatter([neighbour_2[0]],[neighbour_2[1]], color='black')
    ax.annotate('label: 2\nconfidence:  0.9', xy=neighbour_2, 
                xytext=(neighbour_2[0] + 2 * annotation_offset_x, neighbour_2[1] - 2 * annotation_offset_y), 
                arrowprops=dict(fc='w', arrowstyle="-|>"), fontsize=fontsize,fontweight="bold")
    
    
    ax.scatter([neighbour_3[0]],[neighbour_3[1]], color='pink')
    ax.annotate('label: 3\nconfidence:  0.67', xy=neighbour_3, 
                xytext=(neighbour_3[0] + annotation_offset_x, neighbour_3[1] + annotation_offset_y), 
                arrowprops=dict(fc='w', arrowstyle="-|>"), fontsize=fontsize,fontweight="bold")
   
    ax.scatter([neighbour_4[0]],[neighbour_4[1]], color='red')
    ax.annotate('label: 4\nconfidence:  0.74', xy=neighbour_4, 
                xytext=(neighbour_4[0], neighbour_4[1] + annotation_offset_y), 
                arrowprops=dict(fc='w', arrowstyle="-|>"), fontsize=fontsize,fontweight="bold")
    
    # block box
    ax.plot([-1,1], [-1,-1], 'r-')
    ax.plot([-1,1], [1,1], 'r-')
    
    ax.plot([-1,-1], [-1,1], 'r-')
    ax.plot([1,1], [-1,1], 'r-')
    
    # binary split
    
    ax.scatter([central_point[0]],[central_point[1]], color='blue')
    ax.annotate('label: 0\nconfidence:  0.65', xy=central_point, 
                xytext=(central_point[0] + annotation_offset_x, central_point[1] + annotation_offset_y), 
                arrowprops=dict(fc='w', arrowstyle="-|>"), fontsize=fontsize, fontweight="bold")
   
    ax.plot([-1,1], [0,0], 'r-')
    ax.plot([0,0], [-1,1], 'r-')
    
    # sub-blocks pixels
    ax.scatter([0.5,-0.5,-0.5,0.5], [0.5,-0.5,0.5,-0.5], color='green')
    
    """
    # confidence split
    ax.scatter([central_point[0]],[central_point[1]], color='blue')
    ax.annotate('label: 0\nconfidence:  0.65', xy=central_point, 
                xytext=(central_point[0] - 6 * annotation_offset_x, central_point[1] - 6.5 * annotation_offset_y), 
                arrowprops=dict(fc='w', arrowstyle="-|>"), fontsize=fontsize,fontweight="bold")
    
    ax.plot([-0.4,-0.4], [-1,1], 'r-')
    ax.plot([0.5,0.5], [-1,1], 'r-')
    
    ax.plot([-1,1], [-0.4,-0.4], 'r-')
    ax.plot([-1,1], [0.5,0.5], 'r-')
    # sub-blocks pixels
    ax.scatter([-0.7, 0.05, 0.75], [-0.7,-0.7,-0.7], color='green')
    ax.scatter([-0.7, 0.05, 0.75], [0.05,0.05,0.05], color='green')
    ax.scatter([-0.7, 0.05, 0.75], [0.75,0.75,0.75], color='green')
    """
    
    ax.plot()
    plt.show()    

def show_bilinear_interpolation():
    
    ax = plt.figure().add_subplot(projection='3d')
    X = np.linspace(0, 1, 10)
    Y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(X, Y)
    
    z = np.random.random(4)
    
    
    Z = (X + Y) * 0
    ax.plot_surface(X, Y, Z, alpha=0.3, color='red')
    ax.scatter([0, 0, 1, 1], [0, 1, 0, 1], [0,0, 0, 0], color='red')
    ax.plot([0, 0], [0, 0], [0, z[0]], 'r--')
    ax.plot([0, 0], [1, 1], [0, z[1]], 'r--')
    ax.plot([1, 1], [0, 0], [0, z[2]], 'r--')
    ax.plot([1, 1], [1, 1], [0, z[3]], 'r--')
    
    ax.scatter([0, 0, 1, 1], [0, 1, 0, 1], z, c='red')
   
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    X = np.linspace(0, 1, 10)
    Y = np.linspace(0, 1, 10)
   
    Z = interpolate.griddata((x, y), z, (X[None, :], Y[:, None]), method='cubic')
    X, Y = np.meshgrid(X, Y)
   
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', alpha=0.3)
    ax.scatter(X[5][5], Y[5][5], Z[5][5], c='black') 
    ax.scatter(X[5][5], Y[5][5], 0, c='black') 
    ax.plot([X[5][5], X[5][5]], [Y[5][5], Y[5][5]], [0, Z[5][5]], '--', color='black')
    
    ax.axis('off')
    plt.show()
    

def compute_dbm_confidence_interpolation():
    X_train, X_test, Y_train, Y_test = import_data()
    
    resolution = 50
    classifier = import_classifier()
    dbm = DBM(classifier)
    
    X2d_train, X2d_test = import_2d_data()
        
    dbm.generate_boundary_map(X_train,
                              X_test,
                              X2d_train,
                              X2d_test,
                              load_folder=os.path.join("tmp", "MNIST", "DBM"),
                              resolution=10
                              )

    img, img_confidence, _ = dbm._get_img_dbm_fast_confidence_interpolation_strategy(resolution, initial_resolution=32)

    with open(str(resolution) + "_img.npy", "wb") as f:
        np.save(f, img)
    with open(str(resolution) + "_conf.npy", "wb") as f:
        np.save(f, img_confidence)
        
    # img, img_confidence, _ = dbm._get_img_dbm_(resolution)
    # with open(str(resolution) + "_true_img.npy", "wb") as f:
    #     np.save(f, img)
    # with open(str(resolution) + "_true_conf.npy", "wb") as f:
    #     np.save(f, img_confidence)


def test4():
    res = 50
    
    true_img_path = f"experiments/results/MNIST/DBM/t-SNE/none/img/{res}.npy"
    true_img_conf_path = f"experiments/results/MNIST/DBM/t-SNE/none/confidence/{res}.npy"
    
    # true_img_path = f"{res}_true_img.npy"
    # true_img_conf_path = f"{res}_true_conf.npy"

    
    with open(f"{res}_img.npy", "rb") as f:
        interpolated_img = np.load(f)
    with open(true_img_path, "rb") as f:
        ground_truth = np.load(f)
    
    err = 0
    for i in range(res):
        for j in range(res):
            if interpolated_img[i][j] != ground_truth[i][j]:
                err += 1
    print(err)
    print(100 * (err / (res **2)))


    with open(f"{res}_conf.npy", "rb") as f:
        interpolated_img = np.load(f)
    with open(true_img_conf_path, "rb") as f:
        ground_truth = np.load(f)
    
    print(np.sum((ground_truth - interpolated_img) ** 2) / np.sum(ground_truth ** 2))
    
    
def train_mnist_classifier():
    X_train, X_test, Y_train, Y_test = import_data()
    classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax,
                              kernel_initializer=tf.keras.initializers.HeUniform(42))
    ])

    classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classifier.fit(X_train, Y_train, epochs=20, batch_size=32, shuffle=False)

    classifier.evaluate(X_test, Y_test)
    classifier.save(os.path.join("tmp", "MNIST", "classifier"), save_format="tf")

#-------------------------------------------------------------------------------------------
from sklearn.metrics import cohen_kappa_score
from PIL import Image


DATA_FOLDER = os.path.join("data", "proto_simp")
TEST_DATA_FOLDER = "test"
TRAIN_DATA_FOLDER = "train"

def import_feature_extractor():
    # upload a feature extractor
    # !!! the feature extractor encoder and decoder must be tf.keras.models.Model !!!
    # !!! change the next line with the path to your feature extractor !!!
    feature_extractor_path = os.path.join("AuNN")
    encoder_path = os.path.join(feature_extractor_path, "encoder")
    decoder_path = os.path.join(feature_extractor_path, "decoder")
    try:
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)
        return encoder, decoder
    except:
        raise Exception("Error when loading models")


def import_from_folder(folder, true_labels: bool = False):
    files = os.listdir(folder)
    files.sort()
    file_list = [os.path.join(folder, file) for file in files if file.endswith(".png")]
    X = np.array([np.array(Image.open(fname)) for fname in file_list])
    
    if true_labels:
        Y = np.array([int(file.split("_")[0]) for file in files if file.endswith(".png")])
    else:
        with open(os.path.join(folder, "y.npy"), "rb") as f:
            Y = np.load(f)
    
    # the labels are 1 unit wrong (e.g. the image shows a 9 whereas the label is 10)
    Y -= 1
    return X, Y

def import_parasites_data(true_labels: bool = False):
    
    train_dir = os.path.join(DATA_FOLDER, TRAIN_DATA_FOLDER)
    test_dir = os.path.join(DATA_FOLDER, TEST_DATA_FOLDER)

    X_train, Y_train = import_from_folder(train_dir, true_labels)
    X_test, Y_test = import_from_folder(test_dir)

    print(f"X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"X_test {X_test.shape}, Y_test {Y_test.shape}")

    return X_train.astype('float32') / 255, X_test.astype('float32') / 255, Y_train, Y_test


def import_parasites_2d_data(projection='t-SNE'):
    try:
        # upload the 2D projection of the data
        with open(os.path.join("tmp", "data_proto_simp", "DBM", projection, "train_2d.npy"), "rb") as f:
            X2d_train = np.load(f)
        with open(os.path.join("tmp", "data_proto_simp", "DBM", projection, "test_2d.npy"), "rb") as f:
            X2d_test = np.load(f)
        return X2d_train, X2d_test
    except:
        print("No 2d data projection found for projection " + projection)
        return None, None

def make_parasites_classifier(X_train, X_test, Y_train, Y_test):
    # make a classifier
    num_classes = len(np.unique(np.concatenate((Y_train, Y_test))))
    classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax, kernel_initializer=tf.keras.initializers.HeUniform(42))
    ])
    classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classifier.fit(X_train, Y_train, epochs=20, shuffle=False, validation_split=0.2)
    classifier.evaluate(X_test, Y_test)
    Y_pred = classifier.predict(X_test, verbose=0).argmax(axis=-1)
    
    kappa_score = cohen_kappa_score(Y_test, Y_pred)
    print(f"Kappa score: {kappa_score:.4f}")
    
    classifier.save(os.path.join("tmp", "data_proto_simp", "classifier"), save_format="tf")
    

def DBM_usage_example_GUI_with_feature_extraction():
    # import the dataset
    X_train, X_test, Y_train, Y_test = import_parasites_data(true_labels = False)

    # extract features
    feature_extractor, feature_decoder = import_feature_extractor()
    print("Extracting features using the feature extractor...")
    X_train_features = feature_extractor.predict(X_train)  # type: ignore
    X_test_features = feature_extractor.predict(X_test)  # type: ignore

    # getting the classifier
    classifier = tf.keras.models.load_model(os.path.join("tmp", "data_proto_simp", "classifier")) 
   
    # create the DBM
    dbm = DBM(classifier=classifier)

    # import 2d data
    X2d_train, X2d_test = import_parasites_2d_data(projection='t-SNE')
   
    # use the DBM to get the decision boundary map, if you don't have the 2D projection of the data
    # the DBM will get it for you, you just need to specify the projection method you would like to use (t-SNE, PCA or UMAP)
    img, img_confidence, encoded_training_data, encoded_testing_data = dbm.generate_boundary_map(X_train_features,
                                                                                                 X_test_features,
                                                                                                 X2d_train=X2d_train,
                                                                                                 X2d_test=X2d_test,
                                                                                                 resolution=256,
                                                                                                 load_folder=os.path.join("tmp", "data_proto_simp", "DBM"),
                                                                                                 projection="t-SNE",
                                                                                                 )

    dbm_plotter_gui = DBMPlotterGUI(dbm_model=dbm,
                                    img=img,
                                    img_confidence=img_confidence,
                                    encoded_train=encoded_training_data,
                                    encoded_test=encoded_testing_data,
                                    X_train=X_train,
                                    Y_train=Y_train,
                                    X_test=X_test,
                                    Y_test=Y_test,
                                    # this is the folder where the DBM will save the changes in data the user makes
                                    save_folder=os.path.join("tmp", "data_proto_simp", "DBM"),
                                    projection_technique="t-SNE",
                                    helper_decoder=feature_decoder,
                                    X_train_latent=X_train_features,
                                    X_test_latent=X_test_features
                                    )
    dbm_plotter_gui.start()


DBM_usage_example_GUI_with_feature_extraction()

#show_bilinear_interpolation()
#show_grid()
#show_fig_3_10()
#show_img("/Users/cristiangrosu/Desktop/code_repo/MasterThesis2023/experiments/results/MNIST/DBM/t-SNE/none/img/250.npy")
#import_folder_dataset("./data/parasites_focus_plane_divided/larvas/resized")
#test()
#show_errors()
#test3()
#test_interpolation()
#train_mnist_classifier()

#compute_dbm_confidence_interpolation()
#test4()

