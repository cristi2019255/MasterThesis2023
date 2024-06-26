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
from experiments import resolutions_run_times, compute_errors, resolutions_experiment_plot, errors_plot, compute_confidence_errors, compute_confidence_images, confidence_errors_plot
from experiments.scripts.compute_errors import compute_confidence_errors_for_confidence_interpolation
from experiments.scripts.experiment_hyperparameters import hyperparam_run_times, plot_hyperparameter_label_errors, plot_hyperparameter_runtimes
from experiments.scripts.plotter import confidence_errors_plot_for_confidence_interpolation

# ---------------------------------------------------
# INSTRUCTIONS:
#
# WARNING: The experiments need to be run in the following order: 0, 1, 2, 3, 4, ...
#
# EXPERIMENT 0: Get the best value of the parameter B
#
# EXPERIMENT 1: (Run the experiments for resolutions with a specific fast decoding strategy)
#     1. Set the FAST_DECODING_STRATEGY in the experiments/scripts/experiment_resolutions.py file
#     1.1. (Optional) Set the RESOLUTION_RANGE in the experiments/scripts/experiment_resolutions.py file
#     2. Run resolutions_run_times() from this file
#     2.1 WARNING: If the experiment was already run, the results will not be overwritten
#     2.2 WARNING: If you want to run the experiment again, please delete the folder: experiments/results/{dataset_name}/{dbm_strategy}/({projection})/
#     2.3 WARNING: This experiment is time consuming (it can take up to 1 hour) :))
# OUTPUT: The results will be saved in the folder experiments/results/{dataset_name}/{dbm_strategy}/({projection})/{FAST_DECODING_STRATEGY}/
#
# IMPORTANT: Repeat EXPERIMENT 1 for all the fast decoding strategies
#
# EXPERIMENT 2: (Compute a plot to compare the fast decoding strategies run times)
#     1. Set the FOLDER to the corresponding folder for your dataset, dbm strategy and projection
#     2. Run resolutions_experiment_plot(folder=FOLDER) from this file
# OUTPUT: The plot will be saved in the folder experiments/results/{dataset_name}/{dbm_strategy}/({projection})/resolutions_experiment.png

# EXPERIMENT 3: (Compute the errors for the different resolutions)
#     1. Set the FOLDER to the corresponding folder for your dataset, dbm strategy and projection
#     2. Run compute_errors(folder=FOLDER) from this file
# OUTPUT: The errors results will be saved for each fast decoding strategy in the folder:
#         experiments/results/{dataset_name}/{dbm_strategy}/({projection})/{FAST_DECODING_STRATEGY}/errors_results.txt
#
# EXPERIMENT 4: (Compute a plot to compare the errors for the different resolutions)
#     1. Set the FOLDER to the corresponding folder for your dataset, dbm strategy and projection
#     2. Run errors_plot(folder=FOLDER) from this file
# OUTPUT: The plot will be saved in the folder experiments/results/{dataset_name}/{dbm_strategy}/({projection})/errors_experiment.png
#
# DEFINITION: EXPERIMENT 1-4 is called a EXPERIMENT PROJECTION
#
# EXPERIMENT 5: Run EXPERIMENT PROJECTION for all the projections (t-SNE, PCA, UMAP)
#
# DEFINITION: EXPERIMENT 5 is called a EXPERIMENT DBM STRATEGY
# EXPERIMENT 6: Run EXPERIMENT DBM STRATEGY for all the dbm strategies (DBM, SDBM)
#
# DEFINITION: EXPERIMENT 6 is called a EXPERIMENT DATASET
# EXPERIMENT 7: Run EXPERIMENT DATASET for all the datasets (MNIST, Fashion-MNIST, CIFAR-10, [placeholder for parasites dataset])
#
# EXPERIMENT 8: (Compute the confidence images for the different interpolation methods)
#        1. Set the FOLDER to the corresponding folder for your dataset, dbm strategy, projection, fast decoding strategy
#        2. Choose an interpolation method from the following list: 'linear', 'nearest', 'cubic'
#        3. Run compute_confidence_images(folder=FOLDER, interpolation_method={interpolation_method}, resolutions=[250, 500, 1000]) from this file
#        4. Repeat steps 2 and 3 for all the interpolation methods
# OUTPUT: The confidence images will be saved in the folder: experiments/results/{dataset_name}/{dbm_strategy}/({projection})/{FAST_DECODING_STRATEGY}/confidence_images/{interpolation_method}/
#
# EXPERIMENT 9: (Compute errors for the different interpolation methods)
#        1. Set the FOLDER to the corresponding folder for your dataset, dbm strategy, projection, fast decoding strategy and interpolation method
#        2. Run compute_confidence_errors(folder=FOLDER, interpolation_method={interpolation_method}) from this file
# OUTPUT: The confidence errors will be saved in the folder: experiments/results/{dataset_name}/{dbm_strategy}/({projection})/{FAST_DECODING_STRATEGY}/confidence_errors.txt
#
# EXPERIMENT 10: (Compute a plot to compare the errors for the different interpolation methods)
#        1. Set the FOLDER to the corresponding folder for your dataset, dbm strategy, projection, fast decoding strategy
#        2. Run confidence_errors_plot(folder=FOLDER) from this file
# OUTPUT: The plot will be saved in the folder experiments/results/{dataset_name}/{dbm_strategy}/({projection})/{FAST_DECODING_STRATEGY}/confidence_errors.png
#
# EXPERIMENT 11: Repeat EXPERIMENT 8-10 for all the fast decoding strategies, for all the projections, for all the dbm strategies, for all the datasets
#
# ---------------------------------------------------


#FOLDER = os.path.join("experiments", "results", "MNIST", "SDBM", "ssnp")
#FOLDER = os.path.join("experiments", "results", "MNIST", "SDBM", "autoencoder")
#FOLDER = os.path.join("experiments", "results", "MNIST", "DBM", "t-SNE")
#FOLDER = os.path.join("experiments", "results", "MNIST", "DBM", "UMAP")
#FOLDER = os.path.join("experiments", "results", "MNIST", "DBM", "PCA")

# ---------------------------------------------------
#                 RUN THE EXPERIMENT(S)
# ---------------------------------------------------
if __name__ == "__main__":
    
    # ---------------------------------------------------
    #                 EXPERIMENT 0
    # ---------------------------------------------------
    # hyperparam_run_times()
    # plot_hyperparameter_runtimes()
    # plot_hyperparameter_label_errors()
    
    # ---------------------------------------------------
    #                 EXPERIMENT 1-4-5-6-7
    # ---------------------------------------------------
    resolutions_run_times()
    #resolutions_experiment_plot(folder=FOLDER)
    #compute_errors(folder=FOLDER)
    #errors_plot(folder=FOLDER)

    # ---------------------------------------------------
    #                 EXPERIMENT 8-10
    # ---------------------------------------------------
    # compute_confidence_images(folder=os.path.join(FOLDER, "confidence_split"), interpolation_method='nearest', resolutions=[250, 500, 750, 1000, 1250, 1500, 1750, 1950])
    # compute_confidence_images(folder=os.path.join(FOLDER, "binary_split"), interpolation_method='nearest', resolutions=[250, 500, 750, 1000, 1250, 1500, 1750, 1950])
    # compute_confidence_images(folder=os.path.join(FOLDER, "confidence_split"), interpolation_method='linear', resolutions=[250, 500, 750, 1000, 1250, 1500, 1750, 1950])
    # compute_confidence_images(folder=os.path.join(FOLDER, "binary_split"), interpolation_method='linear', resolutions=[250, 500, 750, 1000, 1250, 1500, 1750, 1950])
    # compute_confidence_images(folder=os.path.join(FOLDER, "confidence_split"), interpolation_method='cubic', resolutions=[250, 500, 750, 1000, 1250, 1500, 1750, 1950])
    # compute_confidence_images(folder=os.path.join(FOLDER, "binary_split"), interpolation_method='cubic', resolutions=[250, 500, 750, 1000, 1250, 1500, 1750, 1950])

    # compute_confidence_errors(folder=FOLDER, interpolation_method='nearest')
    # compute_confidence_errors(folder=FOLDER, interpolation_method='linear')
    # compute_confidence_errors(folder=FOLDER, interpolation_method='cubic')

    # confidence_errors_plot(folder=FOLDER)

    # compute_confidence_errors_for_confidence_interpolation(folder=FOLDER)
    # confidence_errors_plot_for_confidence_interpolation(folder=FOLDER)
