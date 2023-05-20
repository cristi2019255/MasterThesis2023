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
from .utils import compute_error

RESULTS_FOLDER = os.path.join("experiments", "results", "MNIST", "DBM", "t-SNE")
ERRORS_FILE_NAME = "errors_results.txt"

def compute_errors(folder=RESULTS_FOLDER):
    strategies_folders = os.listdir(folder)
    ground_truth_img_folder = os.path.join(folder, "none", "img")
    for dir in strategies_folders:
        if dir == "none" or os.path.isfile(os.path.join(folder, dir)):
            continue
        
        errors_file_path = os.path.join(folder, dir, ERRORS_FILE_NAME)
        if os.path.isfile(errors_file_path):
            print("WARNING: The experiment was already run. If you want to run it again, please delete the file: ", errors_file_path)
            continue
        
        with open(errors_file_path, "w") as f:
            f.write("RESOLUTION,ERROR,ERROR RATE\n")
        
        # get all the images in the folder/img
        images_names = os.listdir(os.path.join(folder, dir, "img"))
        for image_name in images_names:
            img_path = os.path.join(folder, dir, "img", image_name)
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
    