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

from src import DBM, SDBM, FAST_DBM_STRATEGIES
from src.decision_boundary_mapper.utils import import_cifar10_dataset, import_fashion_mnist_dataset, import_mnist_dataset
import os
import numpy as np
import tensorflow as tf
import time

SAMPLES_LIMIT = 5000
TRAINING_PERCENTAGE = 0.7
TESTING_PERCENTAGE = 1 - TRAINING_PERCENTAGE

FAST_DECODING_STRATEGY = FAST_DBM_STRATEGIES.BINARY


TMP_FOLDER = "tmp"
DATASET_NAME = "MNIST"

CLASSIFIER_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, "classifier")


DATASET_IMPORTERS = {
    "MNIST": import_mnist_dataset,
    "CIFAR10": import_cifar10_dataset,
    "FASION MNIST": import_fashion_mnist_dataset,
}

DBM_TECHNIQUES = {
    "DBM": DBM,
    "SDBM": SDBM,
}

DBM_TECHNIQUE = "DBM"
PROJECTION = "t-SNE"
TRAIN_2D_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, DBM_TECHNIQUE, PROJECTION, "train_2d.npy")
TEST_2D_PATH = os.path.join(TMP_FOLDER, DATASET_NAME, DBM_TECHNIQUE, PROJECTION, "test_2d.npy")


def import_data(dataset_name=DATASET_NAME, samples_limit=SAMPLES_LIMIT, training_percentage=TRAINING_PERCENTAGE):
    # import the dataset
    (X_train, Y_train), (X_test, Y_test) = DATASET_IMPORTERS[dataset_name]()
    # if needed perform some preprocessing
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train, Y_train = X_train[:int(training_percentage * samples_limit)], Y_train[:int(training_percentage * samples_limit)]
    X_test, Y_test = X_test[:int((1 - training_percentage) * samples_limit)], Y_test[:int((1 - training_percentage) * samples_limit)]
    return X_train, X_test, Y_train, Y_test

def import_2d_data(train_2d_path=TRAIN_2D_PATH, test_2d_path=TEST_2D_PATH):
    # upload the 2D projection of the data
    with open(train_2d_path, "rb") as f:
        X2d_train = np.load(f)
    with open(test_2d_path, "rb") as f:
        X2d_test = np.load(f)
    return X2d_train, X2d_test

def import_classifier(classifier_path=CLASSIFIER_PATH):
    # upload a classifier
    classifier = tf.keras.models.load_model(classifier_path)
    return classifier

def import_dbm(dbm_technique=DBM_TECHNIQUE, classifier_path=CLASSIFIER_PATH):
    # upload a DBM
    classifier = import_classifier(classifier_path)
    dbm = DBM_TECHNIQUES[dbm_technique](classifier)
    return dbm

def compute_error(img1, img2, comparing_confidence=False):
    errors = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i, j] != img2[i, j]:
                if comparing_confidence:
                    if abs(img1[i, j] - img2[i, j]) > 0.03:
                        errors += 1
                else:
                    errors += 1

    errors_rate: float = errors / (img1.shape[0] * img1.shape[1]) * 100
    print("Error: ", errors)
    print("Error rate: ", errors_rate, "%")
    return errors, round(errors_rate, 3)


def save_result(path, result):
    with open(path, "wb") as f:
        np.save(f, result)

def is_experiment(experiment_metadata_path):     
    def function_wrapper(func):
        with open(experiment_metadata_path, "a") as f:
            f.write(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")
            f.write(f"Num CPUs Available: {len(tf.config.list_physical_devices('CPU'))}\n")
            f.write(f"Num TPUs Available: {len(tf.config.list_physical_devices('TPU'))}\n")
            f.write("\n\n\n")
        
        print("Performing experiment: ", func.__name__)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} took {end-start} seconds")
            with open(experiment_metadata_path, "a") as f:
                f.write(f"Experiment {func.__name__} took {end-start} seconds\n")
            return result
        
        return wrapper
    return function_wrapper