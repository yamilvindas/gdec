#!/usr/bin/env python3
"""
    This code allows to train a Transformer Encoder using two different
    representations of an audio signal at the same time (multi-modal).

    Options:
    --------
    --parameters_file: str
        Path to a file containing the parameters of the experiment.
        This files are usually located in /hits_signal_learning/parameters_files/sub-folders/
        WARNING: If it is not, pre-computed results are going to be downloaded
        and the model is not going to be trained from scratch
"""
import os
import json
import pickle
import argparse
from tqdm import tqdm

import random

import numpy as np
from datetime import datetime

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from src.utils.tools import string_to_bool, train_val_split_stratified
from src.utils.download_data import download_dataset, download_results_experiment
from src.DataManipulation.ecg_data import EcgCategorization_Multifeature, load_ecg_datasets_heartbeat_categorization
from src.DataManipulation.eeg_data import EEG_EpilepticSeizureRecognition_Multifeature, loadFromHDF5_EEG

# Models
from src.Models.Hybrid.Multimodal_simple_late_fusion import SimpleAttentionLateFusion_Binary, SimpleAttentionLateFusion_FiveClasses
from src.Models.CNNs.time_frequency_simple_CNN import TimeFrequency2DCNN
from src.Models.Transformers.Transformer_Encoder_RawAudioMultiChannelCNN import TransformerClassifierMultichannelCNN

#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#

class Experiment(object):
    def __init__(self, parameters_exp):
        """
            Class that trains a model with different input features using a
            late fusion model as in Vindas et al. (2022)

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

        # Boolean to see if the results should be saved
        if ('save_res' not in parameters_exp):
            parameters_exp['save_res'] = True
        else:
            parameters_exp['save_res'] = string_to_bool(parameters_exp['save_res'])
        self.save_res = parameters_exp['save_res']

        # Feature type
        if ('features_types' not in parameters_exp):
            parameters_exp['features_types'] = ['RawSignal', 'Spectrogram']
        self.features_types = parameters_exp['features_types']

        # Dataset type
        if ('dataset_type' not in parameters_exp):
            parameters_exp['dataset_type'] = 'ECGCategorization'
        self.dataset_type = parameters_exp['dataset_type']
        if ('subdataset' not in parameters_exp):
            if (self.dataset_type.lower() == 'ecgcategorization'):
                parameters_exp['subdataset'] = 'PTB'
            else:
                parameters_exp['subdataset'] = None
        self.subdataset = parameters_exp['subdataset']

        # Loading the params of the exps which trained the models that we are
        # going to load
        # Model 1
        if ('params_exp_file_1' not in parameters_exp):
            raise ValueError("The parameters of the experiment which allowed to obtain the model 1 are required")
        else:
            with open(parameters_exp['params_exp_file_1'], 'rb') as pf:
                self.parameters_exp_1 = pickle.load(pf)
        # Model 2
        if ('params_exp_file_2' not in parameters_exp):
            raise ValueError("The parameters of the experiment which allowed to obtain the model 2 are required")
        else:
            with open(parameters_exp['params_exp_file_2'], 'rb') as pf:
                self.parameters_exp_2 = pickle.load(pf)

        # Getting the names of the files allowing to load the weights of the trained models
        # Model 1
        if ('model_weights_file_1' not in parameters_exp):
            raise ValueError("The weights of model 1 are required")
        else:
            self.model_weights_file_1 = parameters_exp['model_weights_file_1']
        # Model 2
        if ('model_weights_file_2' not in parameters_exp):
            raise ValueError("The weights of model 2 are required")
        else:
            self.model_weights_file_2 = parameters_exp['model_weights_file_2']

        # Training params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = parameters_exp['lr']
        self.nb_repetitions = parameters_exp['nb_repetitions']
        self.weight_decay = parameters_exp['weight_decay']
        self.batch_size_train = parameters_exp['batch_size_train']
        self.batch_size_test = parameters_exp['batch_size_test']
        self.nb_epochs = parameters_exp['nb_epochs']
        self.criterion = None
        if (parameters_exp['loss_function'].lower() == 'ce'):
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('Loss function {} is not valid'.format(parameters_exp['loss_function']))

        # Creation of val dataset ?
        if ('separate_val_ds' not in parameters_exp):
            parameters_exp['separate_val_ds'] = True
            # parameters_exp['separate_val_ds'] = False
        self.separate_val_ds = parameters_exp['separate_val_ds']

        # Compute class weights parameter
        if ('compute_class_weights' not in parameters_exp):
            parameters_exp['compute_class_weights'] = False
        self.compute_class_weights = parameters_exp['compute_class_weights']

        # Percentage of samples to keep
        if ('percentage_samples_keep' not in parameters_exp):
            parameters_exp['percentage_samples_keep'] = 0.1
        self.percentage_samples_keep = parameters_exp['percentage_samples_keep']

        # Dataset loading
        if (self.dataset_type.lower()=='ecgcategorization'):
            # Download dataset if it has not been done
            download_dataset(dataset_name='ecg_categorization', local_data_directory='../../../data/')
            # Add channel dim
            if ('add_channel_dim' not in parameters_exp):
                parameters_exp['add_channel_dim'] = False
            else:
                parameters_exp['add_channel_dim'] = string_to_bool(parameters_exp['add_channel_dim'])
            self.add_channel_dim = parameters_exp['add_channel_dim']
            self.parameters_exp = parameters_exp

            # Loading the data
            datasets = load_ecg_datasets_heartbeat_categorization(self.parameters_exp['dataset_folder'])
            if (self.subdataset is not None):
                if (self.subdataset.lower() == 'ptb'):
                    self.nb_classes = 2
                    self.training_data = datasets['PTB']['Train']
                    self.testing_data = datasets['PTB']['Test']
                elif (self.subdataset.lower() == 'mitbih'):
                    self.nb_classes = 5
                    self.training_data = datasets['MITBIH']['Train']
                    self.testing_data = datasets['MITBIH']['Test']
                else:
                    raise NotImplementedError("Sudataset {} of EcgCategorization is not implemented".format(self.subdataset))
                parameters_exp['nb_classes'] = self.nb_classes

            # Splitting the train data into train and validation
            if (self.separate_val_ds):
                train_val_splits = train_val_split_stratified(self.training_data, n_splits=1, test_size=0.2)[0]
                self.training_data, self.val_data = train_val_splits['Train'], train_val_splits['Validation']

            # Keeping only a part of the training data
            if (self.percentage_samples_keep < 1):
                new_training_data = {}
                new_training_data_id = 0
                nb_samples_keep = int(self.percentage_samples_keep*len(self.training_data))
                for i in range(nb_samples_keep):
                    new_training_data[new_training_data_id] = self.training_data[i]
                    new_training_data_id += 1
                self.training_data = new_training_data

            # Creating the pytorch datasets
            self.train_ds = EcgCategorization_Multifeature(
                                                data=self.training_data,
                                                features_types=self.features_types,
                                                add_channel_dim=self.add_channel_dim,
                                                params=parameters_exp
                                             )
            if (self.separate_val_ds):
                self.val_ds = EcgCategorization_Multifeature(
                                                    data=self.val_data,
                                                    features_types=self.features_types,
                                                    add_channel_dim=self.add_channel_dim,
                                                    params=parameters_exp
                                                 )
            self.test_ds = EcgCategorization_Multifeature(
                                                    data=self.testing_data,
                                                    features_types=self.features_types,
                                                    add_channel_dim=self.add_channel_dim,
                                                    params=parameters_exp
                                                 )
        elif (self.dataset_type.lower()=='eegepilepticseizure'):
            # Download dataset if it has not been done
            download_dataset(dataset_name='ESR', local_data_directory='../../../data/')
            # Parameters for binarization of the dataset
            if ('binarizeDS' not in parameters_exp):
                parameters_exp['binarizeDS'] = True
            # Add channel dim
            if ('add_channel_dim' not in parameters_exp):
                parameters_exp['add_channel_dim'] = False
            else:
                parameters_exp['add_channel_dim'] = string_to_bool(parameters_exp['add_channel_dim'])
            self.add_channel_dim = parameters_exp['add_channel_dim']
            self.parameters_exp = parameters_exp

            # Loading the data
            hdf5_file_path = parameters_exp['dataset_folder'] + '/data.hdf5'
            self.training_data, self.testing_data = loadFromHDF5_EEG(hdf5_file_path)
            if (self.parameters_exp['binarizeDS']):
                self.nb_classes = 2
            else:
                self.nb_classes = 5
            parameters_exp['nb_classes'] = self.nb_classes

            # Splitting the train data into train and validation
            if (self.separate_val_ds):
                train_val_splits = train_val_split_stratified(self.training_data, n_splits=1, test_size=0.2)[0]
                self.training_data, self.val_data = train_val_splits['Train'], train_val_splits['Validation']

            # Keeping only a part of the training data
            if (self.percentage_samples_keep < 1):
                new_training_data = {}
                new_training_data_id = 0
                nb_samples_keep = int(self.percentage_samples_keep*len(self.training_data))
                for i in range(nb_samples_keep):
                    new_training_data[new_training_data_id] = self.training_data[i]
                    new_training_data_id += 1
                self.training_data = new_training_data

            # Creating the pytorch datasets
            self.train_ds = EEG_EpilepticSeizureRecognition_Multifeature(
                                                data=self.training_data,
                                                features_types=self.features_types,
                                                add_channel_dim=self.add_channel_dim,
                                                params=parameters_exp
                                             )
            if (self.separate_val_ds):
                self.val_ds = EEG_EpilepticSeizureRecognition_Multifeature(
                                                    data=self.val_data,
                                                    features_types=self.features_types,
                                                    add_channel_dim=self.add_channel_dim,
                                                    params=parameters_exp
                                                 )
            self.test_ds = EEG_EpilepticSeizureRecognition_Multifeature(
                                                data=self.testing_data,
                                                features_types=self.features_types,
                                                add_channel_dim=self.add_channel_dim,
                                                params=parameters_exp
                                             )
        else:
            raise ValueError('Dataset type {} is not supported'.format(self.dataset_type))
        print("Number of samples in the training dataset: ", len(self.train_ds))
        if (self.separate_val_ds):
            print("Number of samples in the validation dataset: ", len(self.val_ds))
        print("Number of samples in the testing dataset: ", len(self.test_ds))

        # Determining the audio shape for the selected time-frequency representation
        features, label = self.train_ds[0]
        self.audio_feature_shapes = {}
        for feature_type in features:
            self.audio_feature_shapes[feature_type] = features[feature_type].shape
        print("Shape of the used representations: {}".format(self.audio_feature_shapes))

        # Parameters of the exp
        self.parameters_exp = parameters_exp

    def dataloadersCreation(self):
        """
            Create the train and test dataloader necessary to train and test a
            CNN classification model
        """
        # Training set
        train_indices = list(range(0, len(self.train_ds)))
        train_sampler = SubsetRandomSampler(train_indices)
        # train_sampler = SequentialSampler(train_indices)
        self.train_loader = torch.utils.data.DataLoader(self.train_ds,\
                                                       batch_size=self.batch_size_train,\
                                                       sampler=train_sampler)

        # Validation set
        if (self.separate_val_ds):
            val_indices = list(range(0, len(self.val_ds)))
            val_sampler = SubsetRandomSampler(val_indices)
            # val_sampler = SequentialSampler(val_indices)
            self.val_loader = torch.utils.data.DataLoader(self.val_ds,\
                                                           batch_size=self.batch_size_train,\
                                                           sampler=val_sampler)

        # Testing set
        test_indices = list(range(0, len(self.test_ds)))
        # test_sampler = SubsetRandomSampler(test_indices)
        test_sampler = SequentialSampler(test_indices)
        self.test_loader = torch.utils.data.DataLoader(self.test_ds,\
                                                       batch_size=self.batch_size_test,\
                                                       sampler=test_sampler)

    def modelCreation(self, parameters_exp):
        """
            Creates a model to be trained on the selected time-frequency
            representation
        """
        # Creating the model
        model_type = parameters_exp['model_type']
        model_to_use = parameters_exp['model_to_use']
        if (model_type.lower() == '2dcnn'):
            if (model_to_use.lower() == 'timefrequency2dcnn'):
                nb_init_filters = parameters_exp['nb_init_filters']
                increase_nb_filters_mode = parameters_exp['increase_nb_filters_mode']
                pooling_mode = parameters_exp['pooling_mode']
                dropout_probability = parameters_exp['dropout_probability']
                model = TimeFrequency2DCNN(nb_init_filters=nb_init_filters,
                                 increase_nb_filters_mode=increase_nb_filters_mode,
                                 pooling_mode=pooling_mode,
                                 dropout_probability=dropout_probability,
                                 input_shape=parameters_exp['audio_feature_shape'],
                                 num_classes=self.nb_classes)
            else:
                raise ValueError("Model to use {} is not valid".format(model_to_use))


        elif (model_type.lower() == 'transformer'):
            if (model_to_use.lower() == 'rawaudiomultichannelcnn'):
                print("=======> USING RAWAUDIOMULTICHANNELCNN TRANSFORMER\n")
                d_model = parameters_exp['d_model']
                model = model = TransformerClassifierMultichannelCNN(
                                                                parameters_exp['in_channels'],
                                                                parameters_exp['nhead'],
                                                                parameters_exp['d_hid'],
                                                                parameters_exp['nlayers'],
                                                                parameters_exp['dropout'],
                                                                parameters_exp['nb_features_projection'],
                                                                parameters_exp['d_model'],
                                                                self.nb_classes,
                                                                parameters_exp['classification_pool'],
                                                                parameters_exp['n_conv_layers']
                                                            )
            else:
                raise ValueError("Transformer type {} is not valid".format(model_to_use))
        else:
            raise ValueError("Model type {} is not valid".format(model_type))

        # Sending the model to the correct device
        return model.to(self.device)


    def load_weights_models(self):
        # Loading the data of a model
        model_data_1 = torch.load(self.model_weights_file_1, map_location=torch.device('cpu'))
        model_data_2 = torch.load(self.model_weights_file_2, map_location=torch.device('cpu'))

        # Loading the weights into the model
        self.model_1.load_state_dict(model_data_1['model_state_dict'])
        self.model_2.load_state_dict(model_data_2['model_state_dict'])
        print("===> Model loaded successfully !")

    def balance_classes_loss(self):
        # Getting the labels for the training set
        y_train = np.array([self.train_ds[sample_id][1] for sample_id in range(len(self.train_ds))])

        # Computing the weights
        if (self.compute_class_weights):
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        else:
            class_weights = np.array([1.0 for _ in range(len(np.unique(y_train)))])
        print("\n\nClass weights: {}\n\n".format(class_weights))
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        class_weights = class_weights.to(self.device)

        # Creatining the new weighthed loss
        if (self.parameters_exp['loss_function'].lower() == 'ce'):
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        elif (self.parameters_exp['loss_function'].lower() == 'gce'):
            self.criterion = GeneralizedCrossEntropy(class_weights=class_weights)
            # raise NotImplementedError("Class weighting it is not implemented for GCE loss function")
        else:
            raise ValueError('Loss function {} is not valid'.format(self.parameters_exp['loss_function']))

    def createOptimizer(self, model_parameters):
        # Creating the optimizer
        # self.optimizer = torch.optim.Adamax(model_parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.lr, weight_decay=self.weight_decay)

        # Creating the learning rate scheduler
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',\
                                                               factor=0.1, patience=10,\
                                                               threshold=1e-4, threshold_mode='rel',\
                                                               cooldown=0, min_lr=0, eps=1e-08, verbose=False)


    def compute_forward_pass(self, batch, epoch_nb, keep_grad=True):
        # Getting the data and the labels
        audio_features, labels = batch
        audio_features, labels = {feature_type: audio_features[feature_type].float().to(self.device) for feature_type in audio_features}, labels.to(self.device)

        # Computing the encodings of each modality
        self.model_1.eval()
        self.model_2.eval()
        with (torch.no_grad()):
            output_1 = F.softmax(self.model_1(audio_features[self.parameters_exp_1['feature_type']]), dim=1)
            output_2 = F.softmax(self.model_2(audio_features[self.parameters_exp_2['feature_type']]), dim=1)

        # Computing the classification loss
        out = self.model(output_1, output_2) # Generate predictions
        classification_loss = self.criterion(out, labels.long()) # Calculate loss
        # Final loss
        loss = classification_loss

        # Getting the HARD predictions
        y_true, y_pred = [], []
        for i in range(len(out)):
            true_class = int(labels[i])
            y_true.append(true_class)
            predicted_class = int(out[i].max(0)[1])
            y_pred.append(predicted_class)

        predictions = {
                        'TrueLabels': y_true,
                        'PredictedLabels': y_pred
                    }

        return loss, predictions


    def single_train(self):
        """
            Trains a model one time during self.nb_epochs epochs
        """
        # TODO
        # Creating the dataloaders
        self.dataloadersCreation()

        # Creating the model
        if (self.dataset_type.lower() == 'ecgcategorization'):
            if (self.subdataset.lower() == 'ptb'):
                self.model = SimpleAttentionLateFusion_Binary()
            elif (self.subdataset.lower() == 'mitbih'):
                self.model = SimpleAttentionLateFusion_FiveClasses()
        elif (self.dataset_type.lower() == 'eegepilepticseizure'):
            if (self.parameters_exp['binarizeDS']):
                self.model = SimpleAttentionLateFusion_Binary()
            else:
                self.model = SimpleAttentionLateFusion_FiveClasses()

        # Creating the optimizer
        self.createOptimizer(self.model.parameters())


        # Data structures for the losses and the predictions
        loss_values = {
                        'Train': [0 for _ in range(self.nb_epochs)],
                        'Val': [0 for _ in range(self.nb_epochs)],
                        'Test': [0 for _ in range(self.nb_epochs)]
                      }
        predictions_results = {}
        for dataset_split in ['Train', 'Val', 'Test']:
            predictions_results[dataset_split] = {}
            for type_labels in ['TrueLabels', 'PredictedLabels']:
                predictions_results[dataset_split][type_labels] =  [[] for _ in range(self.nb_epochs)]

        # Epochs
        for epoch in tqdm(range(self.nb_epochs)):
            # Training
            self.model.train()
            tmp_train_losses = []
            for batch in self.train_loader:
                # Zero the parameters gradients
                self.optimizer.zero_grad()

                # Forward pass
                train_loss, train_predictions = self.compute_forward_pass(batch, epoch, keep_grad=True)
                tmp_train_losses.append(train_loss.detach().data.cpu().numpy())

                # Backward pass for the gradient computation
                train_loss.backward()

                # Updating the weights
                self.optimizer.step()

                # Updating the predictions results of the current epoch
                predictions_results['Train']['TrueLabels'][epoch] += train_predictions['TrueLabels']
                predictions_results['Train']['PredictedLabels'][epoch] += train_predictions['PredictedLabels']
            loss_values['Train'][epoch] = np.mean(tmp_train_losses)

            # Validation
            if (self.separate_val_ds):
                with (torch.no_grad()):
                    self.model.eval()
                    tmp_val_losses = []
                    for batch in self.val_loader:
                        # Forward pass
                        val_loss, val_predictions = self.compute_forward_pass(batch, epoch, keep_grad=False)
                        tmp_val_losses.append(val_loss.detach().data.cpu())

                        # Updating the predictions results of the current epoch
                        predictions_results['Val']['TrueLabels'][epoch] += val_predictions['TrueLabels']
                        predictions_results['Val']['PredictedLabels'][epoch] += val_predictions['PredictedLabels']

                    loss_values['Val'][epoch] = np.mean(tmp_val_losses)
                    # Scheduling
                    self.sched.step(loss_values['Val'][epoch])

            # Testing
            with (torch.no_grad()):
                self.model.eval()
                tmp_test_losses = []
                for batch in self.test_loader:
                    # Forward pass
                    test_loss, test_predictions = self.compute_forward_pass(batch, epoch, keep_grad=False)
                    tmp_test_losses.append(test_loss.detach().data.cpu())

                    # Updating the predictions results of the current epoch
                    predictions_results['Test']['TrueLabels'][epoch] += test_predictions['TrueLabels']
                    predictions_results['Test']['PredictedLabels'][epoch] += test_predictions['PredictedLabels']

                loss_values['Test'][epoch] = np.mean(tmp_test_losses)
                print("================================================================================")
                print("METRICS\n")
                print("\n=======>Train loss at epoch {} is {}".format(epoch, loss_values['Train'][epoch]))
                if (self.separate_val_ds):
                    print("\t\tVal loss at epoch {} is {}".format(epoch, loss_values['Val'][epoch]))
                print("\t\tTest loss at epoch {} is {}".format(epoch, loss_values['Test'][epoch]))
                print("\n=======>Train F1 Score at epoch {} is {}\n".format(epoch, f1_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch], average='macro')))
                if (self.separate_val_ds):
                    print("\t\tVal F1 Score at epoch {} is {}".format(epoch, f1_score(predictions_results['Val']['TrueLabels'][epoch], predictions_results['Val']['PredictedLabels'][epoch], average='micro')))
                print("\t\tTest F1 Score at epoch {} is {}".format(epoch, f1_score(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch], average='micro')))
                print("\n=======>Train accuracy at epoch {} is {}\n".format(epoch, accuracy_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch])))
                if (self.separate_val_ds):
                    print("\t\tVal accuracy at epoch {} is {}".format(epoch, accuracy_score(predictions_results['Val']['TrueLabels'][epoch], predictions_results['Val']['PredictedLabels'][epoch])))
                print("\t\tTest accuracy at epoch {} is {}".format(epoch, accuracy_score(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch])))
                print("\n=======>Train MCC at epoch {} is {}\n".format(epoch, matthews_corrcoef(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch])))
                if (self.separate_val_ds):
                    print("\t\tVal MCC at epoch {} is {}".format(epoch, matthews_corrcoef(predictions_results['Val']['TrueLabels'][epoch], predictions_results['Val']['PredictedLabels'][epoch])))
                print("\t\tTest MCC at epoch {} is {}".format(epoch, matthews_corrcoef(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch])))
                print("================================================================================\n\n")

            # Saving the model and the current results
            current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
            # Model
            if (self.save_res):
                torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': train_loss
                            }, self.results_folder + '/model/chechkpoint_model-{}.pth'.format(self.exp_id))
                # Current results
                checkpoint_results_file = self.results_folder + '/metrics/checkpoint_results-{}_epoch-{}_{}.pth'.format(self.exp_id, epoch, current_datetime)
                intermediate_results = {'Loss': loss_values, 'Predictions': predictions_results}
                with open(checkpoint_results_file, "wb") as fp:   #Pickling
                    pickle.dump(intermediate_results, fp)

        print("\n\n=========> Attention weights <=========")
        with torch.no_grad():
            if (self.dataset_type.lower() == 'ecgcategorization'):
                if (self.subdataset.lower() == 'ptb'):
                    # print("Class 1 weights: ", F.softmax(self.model.attention_weight_class_1, dim=0))
                    print("Class 1 weights: ", self.model.attention_weight_class_1)
                    # print("Class 2 weights: ", F.softmax(self.model.attention_weight_class_2, dim=0))
                    print("Class 2 weights: ", self.model.attention_weight_class_2)
                elif (self.subdataset.lower() == 'mitbih'):
                    # print("Class 1 weights: ", F.softmax(self.model.attention_weight_class_1, dim=0))
                    print("Class 1 weights: ", self.model.attention_weight_class_1)
                    # print("Class 2 weights: ", F.softmax(self.model.attention_weight_class_2, dim=0))
                    print("Class 2 weights: ", self.model.attention_weight_class_2)
                    # print("Class 3 weights: ", F.softmax(self.model.attention_weight_class_3, dim=0))
                    print("Class 3 weights: ", self.model.attention_weight_class_3)
                    # print("Class 4 weights: ", F.softmax(self.model.attention_weight_class_4, dim=0))
                    print("Class 4 weights: ", self.model.attention_weight_class_4)
                    # print("Class 5 weights: ", F.softmax(self.model.attention_weight_class_5, dim=0))
                    print("Class 5 weights: ", self.model.attention_weight_class_5)
            if (self.dataset_type.lower() == 'eegepilepticseizure'):
                if (self.parameters_exp['binarizeDS']):
                    # print("Class 1 weights: ", F.softmax(self.model.attention_weight_class_1, dim=0))
                    print("Class 1 weights: ", self.model.attention_weight_class_1)
                    # print("Class 2 weights: ", F.softmax(self.model.attention_weight_class_2, dim=0))
                    print("Class 2 weights: ", self.model.attention_weight_class_2)
                else:
                    # print("Class 1 weights: ", F.softmax(self.model.attention_weight_class_1, dim=0))
                    print("Class 1 weights: ", self.model.attention_weight_class_1)
                    # print("Class 2 weights: ", F.softmax(self.model.attention_weight_class_2, dim=0))
                    print("Class 2 weights: ", self.model.attention_weight_class_2)
                    # print("Class 3 weights: ", F.softmax(self.model.attention_weight_class_3, dim=0))
                    print("Class 3 weights: ", self.model.attention_weight_class_3)
                    # print("Class 4 weights: ", F.softmax(self.model.attention_weight_class_4, dim=0))
                    print("Class 4 weights: ", self.model.attention_weight_class_4)
                    # print("Class 5 weights: ", F.softmax(self.model.attention_weight_class_5, dim=0))
                    print("Class 5 weights: ", self.model.attention_weight_class_5)

        return {'Loss': loss_values, 'Predictions': predictions_results}


    def holdout_train(self):
        """
            Does a holdout training repeated self.nb_repetitions times
        """
        # Loading the trained models (to use as encoders for eac modality)
        self.model_1 = self.modelCreation(self.parameters_exp_1)
        self.model_2 = self.modelCreation(self.parameters_exp_2)
        self.load_weights_models()

        # Training the simple late fusion classification
        repetitionsResults = {}
        for nb_repetition in range(self.nb_repetitions):
            print("\n\n=======> Repetitions {} <=======".format(nb_repetition))
            # Doing single train
            tmp_results = self.single_train()
            repetitionsResults[nb_repetition] = tmp_results

            # Saving the final model and the results
            if (self.save_res):
                # Model
                torch.save({
                                'model_state_dict': self.model.state_dict(),
                                'model': self.model
                            }, self.results_folder + '/model/final_model-{}_rep-{}.pth'.format(self.exp_id, nb_repetition))
                # Results
                with open(self.results_folder + '/metrics/results_exp-{}_rep-{}.pth'.format(self.exp_id, nb_repetition), "wb") as fp:   #Pickling
                    pickle.dump(tmp_results, fp)

        if (self.save_res):
            # Saving the results of the different repetitions
            with open(self.results_folder + '/metrics/final_results_all_repetitions.pth', "wb") as fp:   #Pickling
                pickle.dump(repetitionsResults, fp)

    def setResultsFolder(self, results_folder):
        """
            Set the folder where the results are going to be stored
        """
        self.results_folder = results_folder



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
    ap.add_argument('--parameters_file', default=None, help="Parameters for the experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    if (parameters_file is not None):
        with open(parameters_file) as jf:
            parameters_exp = json.load(jf)

    #==========================================================================#
    if (parameters_file is None):
        # Downloading the datasets
        download_dataset(dataset_name='ecg_categorization', local_data_directory='../../../data/')
        download_dataset(dataset_name='ESR', local_data_directory='../../../data/')
        
        # Downloading the models
        print("\n\n===> WARNING: no parameters file was given, insted of running the experiment, we are going to download pre-computed results !\n\n")
        download_results_experiment(experiment_name='Exp-1_LateFusion', local_data_directory='../../../results/')
    else:
        # Creating an instance of the experiment
        exp = Experiment(parameters_exp)

        # Creating directory to save the results
        inc = 0
        current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
        resultsFolder = '../../../results/Experiment_1/' + parameters_exp['exp_id'] + '_' + current_datetime
        while (os.path.isdir(resultsFolder+ '_' + str(inc))):
            inc += 1
        resultsFolder = resultsFolder + '_' + str(inc)
        os.mkdir(resultsFolder)
        exp.setResultsFolder(resultsFolder)
        print("===> Saving the results of the experiment in {}".format(resultsFolder))

        # Creating directories for the trained models, the training and testing metrics
        # and the parameters of the model (i.e. the training parameters and the network
        # architecture)
        os.mkdir(resultsFolder + '/model/')
        os.mkdir(resultsFolder + '/params_exp/')
        os.mkdir(resultsFolder + '/metrics/')

        # Balancing the classes
        exp.balance_classes_loss()

        # Saving the training parameters in the folder of the results
        inc = 0
        parameters_file = resultsFolder + '/params_exp/params_beginning' + '_'
        while (os.path.isfile(parameters_file + str(inc) + '.pth')):
            inc += 1
        parameters_file = parameters_file + str(inc) +'.pth'
        parameters_exp['audio_feature_shape'] = exp.audio_feature_shapes
        with open(parameters_file, "wb") as fp:   #Pickling
            pickle.dump(parameters_exp, fp)

        # Doing holdout evaluation
        exp.holdout_train()

        # Saving the training parameters in the folder of the results
        inc = 0
        parameters_file = resultsFolder + '/params_exp/params' + '_'
        while (os.path.isfile(parameters_file + str(inc) + '.pth')):
            inc += 1
        parameters_file = parameters_file + str(inc) +'.pth'
        parameters_exp['audio_feature_shapes'] = exp.audio_feature_shapes
        if (exp.save_res):
            with open(parameters_file, "wb") as fp:   #Pickling
                pickle.dump(parameters_exp, fp)

    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")


if __name__=="__main__":
    main()
