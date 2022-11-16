#!/usr/bin/env python3
"""
    This code allows to study the influence of the guided training on the
    performances of a multi-feature model

    Options:
    --------
    --parameters_file: str
        Path to a file containing the parameters of the experiment.
        This files are usually located in /hits_signal_learning/parameters_files/sub-folders/
"""
import os
import json
import shutil
import pickle
import argparse
from tqdm import tqdm

import random

import torch
import numpy as np
from datetime import datetime

from src.utils.download_data import download_dataset
from src.Experiments.Experiment_1.training_model_multiple_features import Experiment as Exp1

#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#
class Experiment(object):
    def __init__(self, parameters_exp):
        """
            Class that studies the influence of guided training on multi-feature
            models.

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment:
                    * exp_id: str, name of the experiment.
                    * feature_type: str
                    * ...
        """
        # Defining some attributes of the experiment
        self.exp_id = parameters_exp['exp_id']
        self.results_folder = None

        # Deactivating DEC
        self.use_DEC = False
        parameters_exp['use_DEC'] = self.use_DEC

        # Activating guided training
        self.use_iterated_loss = True
        parameters_exp['use_iterated_loss'] = self.use_iterated_loss

        # Values to use for the importance of the iterated losses
        self.model_apply_iterated_loss = parameters_exp['model_apply_iterated_loss']
        if (self.model_apply_iterated_loss.lower() == 'specencoder') or (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
            self.vals_importance_iterated_loss_spec = parameters_exp['vals_importance_iterated_loss_spec']
        else:
            self.vals_importance_iterated_loss_spec = [0]
        if (self.model_apply_iterated_loss.lower() == 'rawencoder') or (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
            self.vals_importance_iterated_loss_raw = parameters_exp['vals_importance_iterated_loss_raw']
        else:
            self.vals_importance_iterated_loss_raw = [0]

        # Parameters of the exp
        self.parameters_exp = parameters_exp


    def setResultsFolder(self, results_folder):
        """
            Set the folder where the results are going to be stored
        """
        self.results_folder = results_folder

    def run(self):
        """
            Run the experiment
        """
        # Iterating over the importance of the TFR
        results_influence_guided_training = {}
        for importance_iterated_loss_spec in tqdm(self.vals_importance_iterated_loss_spec, desc='\u03B1 loop'):
            self.parameters_exp['importance_iterated_loss_spec'] = importance_iterated_loss_spec
            # Iterating over the importance of the raw signal
            for importance_iterated_loss_raw in tqdm(self.vals_importance_iterated_loss_raw, desc='\u03B2 loop', leave=False):
                print("\n\n\n=======> STARTING Training model with \u03B1 = {} and \u03B2 = {} <=======\n".format(importance_iterated_loss_spec, importance_iterated_loss_raw))
                self.parameters_exp['importance_iterated_loss_raw'] = importance_iterated_loss_raw
                # Creating an instance of experiment 1
                exp1 = Exp1(self.parameters_exp)

                # Balancing the classes
                exp1.balance_classes_loss()

                # Doing one simple training
                tmp_results = exp1.single_train(save_checkpoints=False)

                # Adding the intermediate results to the list of results
                results_influence_guided_training[(importance_iterated_loss_spec, importance_iterated_loss_raw)] = tmp_results

                # Saving intermediate results
                exp1_id = self.exp_id + '_Alpha-{}_'.format(importance_iterated_loss_spec) + 'Beta-{}'.format(importance_iterated_loss_raw)
                with open(self.results_folder + '/metrics/intermediate_results_{}.pth'.format(exp1_id), "wb") as fp:   #Pickling
                    pickle.dump(tmp_results, fp)
                print("\n=======> ENDING Training model with \u03B1 = {} and \u03B2 = {} <=======\n".format(importance_iterated_loss_spec, importance_iterated_loss_raw))
        # Saving the final results
        with open(self.results_folder + '/metrics/final_results_{}.pth'.format(self.exp_id), "wb") as fp:   #Pickling
            pickle.dump(results_influence_guided_training, fp)

#==============================================================================#
#================================ Main Function ================================#
#==============================================================================#
def main():
    print("\n\n==================== Beginning of the experiment ====================\n\n")
    #==========================================================================#
    # Fixing the random seed
    seed = 42
    random.seed(seed) # For reproducibility purposes
    np.random.seed(seed) # For reproducibility purposes
    torch.manual_seed(seed) # For reproducibility purposes
    if torch.cuda.is_available(): # For reproducibility purposes
        torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True) # For reproducibility purposes

    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_parameters_file = "../../../parameters_files/Experiment_2/PTB/concatenation_TFR_and_RawSignal.json"
    ap.add_argument('--parameters_file', default=default_parameters_file, help="Parameters for the experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    with open(parameters_file) as jf:
        parameters_exp = json.load(jf)

    #==========================================================================#
    # Creating an instance of the experiment
    exp = Experiment(parameters_exp)

    # Creating directory to save the results
    inc = 0
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    resultsFolder = '../../../results/Experiment_2/' + parameters_exp['exp_id'] + '_' + current_datetime
    while (os.path.isdir(resultsFolder+ '_' + str(inc))):
        inc += 1
    resultsFolder = resultsFolder + '_' + str(inc)
    os.mkdir(resultsFolder)
    exp.setResultsFolder(resultsFolder)
    print("===> Saving the results of the experiment in {}".format(resultsFolder))

    # Creating directories for the training and testing metrics and the
    # parameters of the experiment (i.e. the training parameters and the network
    # architecture)
    os.mkdir(resultsFolder + '/params_exp/')
    os.mkdir(resultsFolder + '/metrics/')

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params_beginning' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.pth')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.pth'
    with open(parameters_file, "wb") as fp:   #Pickling
        pickle.dump(parameters_exp, fp)

    # Doing holdout evaluation
    exp.run()

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.pth')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.pth'
    with open(parameters_file, "wb") as fp:   #Pickling
        pickle.dump(parameters_exp, fp)

    # Saving the python file containing the network architecture
    if (parameters_exp['model_type'].lower() == 'hybrid'):
        if (parameters_exp['model_to_use'].lower() == 'bimodalcnntransformer_raw+spec'):
            shutil.copy2('../../Models/Hybrid/Transformer_CNN_RawAndSpec.py', resultsFolder + '/params_exp/network_architecture.py')
        else:
            raise ValueError('1D CNN {} is not valid'.format(parameters_exp['model_to_use']))
    else:
        raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))

    # Save the data distribution
    if (parameters_exp['dataset_type'].lower() == 'eegepilepticseizure'):
        shutil.copy2(parameters_exp['dataset_folder'] + '/data.hdf5', resultsFolder + '/params_exp/data.hdf5')


    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()
