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
import pandas as pd
import matplotlib.pyplot as plt

from experiments.scripts.config import CONFIDENCE_ERRORS_RESULTS_FILE_NAME, IMG_ERRORS_RESULTS_FILE_NAME, CONFIDENCE_INTERPOLATION_DECODING_STRATEGY_FOLDER_NAME

RESULTS_FOLDER = os.path.join("experiments", "results", "MNIST", "DBM", "t-SNE")

def resolutions_experiment_plot(folder=RESULTS_FOLDER):
    strategies_folders = os.listdir(folder)
    #strategies_folders = ["none", "confidence_interpolation"]
    
    for dir in strategies_folders:
        if os.path.isfile(os.path.join(folder, dir)):
            continue
        path = os.path.join(folder, dir, "experiment_results.txt")
        df = pd.read_csv(path)
        label = dir
        if dir == "none":
            label = "dummy_dbm"            
        plt.plot(df["RESOLUTION"], df["TIME"], label=label)
    
    plt.title("Resolutions experiment")
    plt.xlabel("Resolution")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig(os.path.join(folder, "resolutions_experiment.png"))
    plt.show()
    
def errors_plot(folder=RESULTS_FOLDER):
    strategies_folders = os.listdir(folder)
    #strategies_folders = ["binary_split"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.title("Errors vs Resolution")
    
    for dir in strategies_folders:
        if dir == "none" or os.path.isfile(os.path.join(folder, dir)):
            continue
        path = os.path.join(folder, dir, IMG_ERRORS_RESULTS_FILE_NAME)
        df = pd.read_csv(path)
        df["RESOLUTION"] = df["RESOLUTION"].astype('int')
        df = df.sort_values(by=["RESOLUTION"])
        ax1.plot(df["RESOLUTION"], df["ERROR"], label=dir)
        ax2.plot(df["RESOLUTION"], df["ERROR RATE"], label=dir)
    
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("Error rate")
    ax2.legend()
    plt.savefig(os.path.join(folder, "errors_experiment.png"))
    
    plt.show()
    
def confidence_errors_plot(folder=RESULTS_FOLDER):
    strategies_folders = os.listdir(folder)
    #strategies_folders = ["none", "confidence_split"]
    interpolation_methods = ["nearest", "linear", "cubic"]
    
    for dir in strategies_folders:
        if dir == "none" or dir=="confidence_interpolation" or os.path.isfile(os.path.join(folder, dir)):
            continue
        
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
        plt.title("Confidence errors vs Resolution")
        for interpolation_method in interpolation_methods:
            path = os.path.join(folder, dir, str(interpolation_method) + "_" + CONFIDENCE_ERRORS_RESULTS_FILE_NAME)
            df = pd.read_csv(path)
            df["RESOLUTION"] = df["RESOLUTION"].astype('int')
            df = df.sort_values(by=["RESOLUTION"])
            ax1.plot(df["RESOLUTION"], df["ERROR"], label=interpolation_method)
            
        ax1.set_xlabel("Resolution")
        ax1.set_ylabel("Error")
        ax1.legend()
        plt.savefig(os.path.join(folder, dir, "confidence_errors_experiment.png"))
        
    #plt.show()
    
def confidence_errors_plot_for_confidence_interpolation(folder=RESULTS_FOLDER):
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
    
    plt.title("Confidence errors vs Resolution")
        
    path = os.path.join(folder, CONFIDENCE_INTERPOLATION_DECODING_STRATEGY_FOLDER_NAME, CONFIDENCE_ERRORS_RESULTS_FILE_NAME)
    df = pd.read_csv(path)
    df["RESOLUTION"] = df["RESOLUTION"].astype('int')
    df = df.sort_values(by=["RESOLUTION"])
    ax1.plot(df["RESOLUTION"], df["ERROR"], label="confidence interpolation strategy")
    
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("Error")
    ax1.legend()
    
    plt.savefig(os.path.join(folder, CONFIDENCE_INTERPOLATION_DECODING_STRATEGY_FOLDER_NAME, "confidence_errors_experiment.png"))
    
    plt.show()
    