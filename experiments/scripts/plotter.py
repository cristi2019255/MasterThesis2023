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

RESULTS_FOLDER = os.path.join("experiments", "results", "MNIST", "DBM", "t-SNE")

def resolutions_experiment_plot(folder=RESULTS_FOLDER):
    strategies_folders = os.listdir(folder)
    
    for dir in strategies_folders:
        path = os.path.join(folder, dir, "experiment_results.txt")
        df = pd.read_csv(path)
        plt.plot(df["RESOLUTION"], df["TIME"], label=dir)
    
    plt.title("Resolutions experiment")
    plt.xlabel("Resolution")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, "resolutions_experiment.png"))
    plt.show()
    
def errors_plot(folder=RESULTS_FOLDER):
    strategies_folders = os.listdir(folder)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.title("Errors vs Resolution")
    
    for dir in strategies_folders:
        if dir == "none":
            continue
        path = os.path.join(folder, dir, "errors_results.txt")
        df = pd.read_csv(path)
        ax1.plot(df["RESOLUTION"], df["ERROR"], label=dir)
        ax2.plot(df["RESOLUTION"], df["ERROR RATE"], label=dir)
    
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("Error rate")
    ax2.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, "errors_experiment.png"))
    
    plt.show()