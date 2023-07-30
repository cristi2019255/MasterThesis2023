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

from .config import CONFIDENCE_FOLDER_NAME, IMG_FOLDER_NAME, INTERPOLATION_FOLDER_NAME, IMG_ERRORS_RESULTS_FILE_NAME, CONFIDENCE_ERRORS_RESULTS_FILE_NAME
from .utils import compute_error

RESULTS_FOLDER = os.path.join("experiments", "results", "MNIST", "DBM", "t-SNE")

def compute_errors(folder=RESULTS_FOLDER):
    strategies_folders = os.listdir(folder)
    ground_truth_img_folder = os.path.join(folder, "none", IMG_FOLDER_NAME)
    for dir in strategies_folders:
        if dir == "none" or os.path.isfile(os.path.join(folder, dir)):
            continue
        
        errors_file_path = os.path.join(folder, dir, IMG_ERRORS_RESULTS_FILE_NAME)
        if os.path.isfile(errors_file_path):
            print("WARNING: The experiment was already run. If you want to run it again, please delete the file: ", errors_file_path)
            continue
        
        print("Computing errors for: ", dir)
        
        with open(errors_file_path, "w") as f:
            f.write("RESOLUTION,ERROR,ERROR RATE\n")
        
        # get all the images in the folder/img
        images_names = os.listdir(os.path.join(folder, dir, IMG_FOLDER_NAME))
        for image_name in images_names:
            img_path = os.path.join(folder, dir, IMG_FOLDER_NAME, image_name)
            ground_truth_img_path = os.path.join(ground_truth_img_folder, image_name)
            resolution = image_name.split(".")[0]
            # getting the images
            with open(img_path, "rb") as f:
                img = np.load(f)
            with open(ground_truth_img_path, "rb") as f:
                ground_truth_img = np.load(f)
            # computing the error
            error, error_rate = compute_error(img, ground_truth_img)  
            line = f"{resolution},{str(error)},{str(error_rate)}\n"
            with open(errors_file_path, "a") as file:
                file.write(line)
    
    print("Errors computed successfully")
    
def compute_confidence_errors(folder=RESULTS_FOLDER, interpolation_method='linear'):
    strategies_folders = os.listdir(folder)
    ground_truth_img_folder = os.path.join(folder, "none", CONFIDENCE_FOLDER_NAME)
    for dir in strategies_folders:
        if dir == "none" or dir=="confidence_interpolation" or os.path.isfile(os.path.join(folder, dir)):
            continue
        
        errors_file_path = os.path.join(folder, dir, interpolation_method + "_" + CONFIDENCE_ERRORS_RESULTS_FILE_NAME)
        #if os.path.isfile(errors_file_path):
        #    print("WARNING: The experiment was already run. If you want to run it again, please delete the file: ", errors_file_path)
        #    continue
        
        with open(errors_file_path, "w") as f:
            f.write("RESOLUTION,ERROR,ERROR RATE\n")
        
        # get all the images in the folder/img
        confidence_names = os.listdir(os.path.join(folder, dir, INTERPOLATION_FOLDER_NAME, interpolation_method))
        for conf_img_name in confidence_names:
            conf_img_path = os.path.join(folder, dir, INTERPOLATION_FOLDER_NAME, interpolation_method, conf_img_name)
            ground_truth_conf_img_path = os.path.join(ground_truth_img_folder, conf_img_name)
            resolution = conf_img_name.split(".")[0]
            # getting the images
            with open(conf_img_path, "rb") as f:
                conf_img = np.load(f)
            with open(ground_truth_conf_img_path, "rb") as f:
                ground_truth_conf_img = np.load(f)
            # computing the error
            error, error_rate = compute_error(conf_img, ground_truth_conf_img, comparing_confidence=True)  
            line = f"{resolution},{str(error)},{str(error_rate)}\n"
            with open(errors_file_path, "a") as file:
                file.write(line)
    
    print("Confidence errors computed successfully")

def compute_confidence_errors_for_confidence_interpolation(folder=RESULTS_FOLDER):
    conf_interpolation_folder = os.path.join(folder, "confidence_interpolation", CONFIDENCE_FOLDER_NAME)
    ground_truth_conf_img_folder = os.path.join(folder, "none", CONFIDENCE_FOLDER_NAME)
    
    errors_file_path = os.path.join(folder, "confidence_interpolation", CONFIDENCE_ERRORS_RESULTS_FILE_NAME)
    with open(errors_file_path, "w") as f:
            f.write("RESOLUTION,ERROR,ERROR RATE\n")
      
    confidence_names = os.listdir(conf_interpolation_folder)
    for conf_img_name in confidence_names:
        conf_img_path = os.path.join(conf_interpolation_folder, conf_img_name)
        ground_truth_conf_img_path = os.path.join(ground_truth_conf_img_folder, conf_img_name)
        with open(conf_img_path, "rb") as f:
            conf_img = np.load(f)
        with open(ground_truth_conf_img_path, "rb") as f:
            ground_truth_conf_img = np.load(f)
        
        resolution = conf_img_name.split(".")[0]   
        error, error_rate = compute_error(conf_img, ground_truth_conf_img, comparing_confidence=True)  
        line = f"{resolution},{str(error)},{str(error_rate)}\n"
        with open(errors_file_path, "a") as file:
            file.write(line)
    
    print("Confidence errors for confidence_interpolation method computed successfully")
