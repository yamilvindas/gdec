#!/usr/bin/env python3
"""
    Plot the results of the experiment

    Options:
    --------
    results_file: str
        Path to the final results file containing the final results over
        the repetitions
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from src.utils.plot_results_experiment_1 import plotMeanLoss, plotMeanMetrics

def load_results_data_exp_2_and_3(results_file):
    """
        Load the results of an experiment (2 or 3) into a dictionary.

        Arguments:
        ----------
        results_file: str
            Path to a pth file containing the results of an experiment.

        Returns:
        --------
        results_dict: dict
            Dictionary containing the results of the experiment over the
            different repetitions and epochs. The structure of this dictionary
            is the following:
            -results_dict is a dict where the keys are the used hyper-parameters
            ((alpha, beta) or gamma).
            -results_dict[used_hyper_params] is a dict with two keys 'Loss' and 'Predictions'.
            -results_dict[used_hyper_params]['Loss'] and results_dict[used_hyper_params]['Predictions']
            are also dictionaries with three keys: 'Train', 'Val', and 'Test'.
            -results_dict[used_hyper_params]['Loss']['Train'] is a list of lenght the
            number of epochs used during training. Each element correspond to the
            loss function at the given epoch.
            -results_dict[used_hyper_params]['Predictions']['Train'] is a dict with
            two keys: 'TrueLabels' and 'PredictedLabels'.
            -results_dict[used_hyper_params]['Predictions']['Train']['TrueLabels'] is
            a list of lenght the number of epochs used during training. Each
            element correspond to a list of size the number of training samples,
            and each element corresponds to the true label of a given sample.
            -results_dict[used_hyper_params]['Predictions']['Train']['PredictedLabels'] is
            a list of lenght the number of epochs used during training. Each
            element correspond to a list of size the number of training samples,
            and each element corresponds to the predicted label of a given sample.
    """
    # Loading the data into a dict
    with open(results_file, "rb") as fp:   # Unpickling
        tmp_dict = pickle.load(fp)

    # Putting the results dictionary in the right format to be able to use the
    # functions of plot_results_experiment_1
    results_dict = {}
    for used_hyper_params in tmp_dict:
        results_dict[used_hyper_params] = {0: tmp_dict[used_hyper_params]}

    return results_dict


#==============================================================================#
#==============================================================================#

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--results_file', required=True, help="File containing the metrics of the experiment", type=str)
    ap.add_argument('--selected_epoch', default=0, help="Particular epoch to compute the different metrics", type=int)
    ap.add_argument('--last_epochs_use', default=1, help="Last epochs to use to compute the mean metrics", type=int)
    ap.add_argument('--plot_curves', default='False', help="Boolean to plot the loss and metrics curves", type=str)
    ap.add_argument('--plot_val_metrics', default='False', help="True if wanted to compute the val metrics and plot it. If there are no validation metrics, this argument should be False", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_file = args['results_file']
    selected_epoch = args['selected_epoch']
    last_epochs_use = args['last_epochs_use']
    plot_curves = args['plot_curves']
    if (plot_curves.lower() == 'true'):
        plot_curves = True
    else:
        plot_curves = False
    plot_val_metrics = args['plot_val_metrics']
    if (plot_val_metrics.lower() == 'true'):
        plot_val_metrics = True
    else:
        plot_val_metrics = False

    #==========================================================================#
    # Loading the data
    results_exps_dict = load_results_data_exp_2_and_3(results_file)

    #==========================================================================#
    for used_hyper_params in results_exps_dict:
        print("\n\n\n\n =======> Plotting the results of {} <=======\n".format(used_hyper_params))
        # Getting the current results dictionary
        results_dict = results_exps_dict[used_hyper_params]

        # Plotting the loss
        if ('Loss' in results_dict[0]):
            plotMeanLoss(results_dict, plot_curves=plot_curves)

        # Plotting the MCC
        plotMeanMetrics(results_dict, metric_type='mcc', selected_epoch=selected_epoch, last_epochs_use=last_epochs_use, plot_curves=plot_curves, plot_val_metrics=plot_val_metrics)

        # Plotting the F1 Score
        plotMeanMetrics(results_dict, metric_type='F1-Score', selected_epoch=selected_epoch, last_epochs_use=last_epochs_use, plot_curves=plot_curves, plot_val_metrics=plot_val_metrics)

        # Plotting the General accuracy
        plotMeanMetrics(results_dict, metric_type='GeneralAccuracy', selected_epoch=selected_epoch, last_epochs_use=last_epochs_use, plot_curves=plot_curves, plot_val_metrics=plot_val_metrics)

        print("\n\n\n")

if __name__=='__main__':
    main()
