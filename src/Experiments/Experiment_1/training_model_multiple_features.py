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
import shutil
import pickle
import argparse
from tqdm import tqdm

import random

import numpy as np
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from labml_nn.optimizers import noam

from src.utils.DEC import DECLoss
from src.utils.tools import string_to_bool, train_val_split_stratified
from src.utils.download_data import download_dataset, download_results_experiment
from src.DataManipulation.ecg_data import EcgCategorization_Multifeature, load_ecg_datasets_heartbeat_categorization
from src.DataManipulation.eeg_data import EEG_EpilepticSeizureRecognition_Multifeature, loadFromHDF5_EEG

# Models
from src.Models.Hybrid.Transformer_CNN_RawAndSpec import TransformerClassifierBimodal_RawAndSpec

#==============================================================================#
#======================== Model for the iterated loss ========================#
#==============================================================================#

class IteratedLossClassifierRawSignal(nn.Module):

    def __init__(
                    self,
                    dim_input_space,
                    num_classes
                ):
        super().__init__()
        # Classification layers
        nb_features_projection = 64
        self.layer_norm_1 = nn.LayerNorm(dim_input_space)
        self.out_1 = nn.Linear(dim_input_space, nb_features_projection)
        self.layer_norm_2 = nn.LayerNorm(nb_features_projection)
        self.out_2 = nn.Linear(nb_features_projection, num_classes)

    def forward(self, src):
        """
        Args:
        """
        # Classification layers
        output = self.out_1(self.layer_norm_1(src))
        output = self.out_2(self.layer_norm_2(output))

        return output

class IteratedLossClassifierSpectrogram(nn.Module):

    def __init__(
                    self,
                    encoding_shape,
                    num_classes,
                    dropout_probability=0.1
                ):
        super().__init__()
        # Classification layers
        in_features = np.prod(encoding_shape)
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes) # 3 out features because we have 3 classes

        # Dropout layer
        self.dropout = nn.Dropout2d(p=dropout_probability)

    def forward(self, src):
        """
        Args:
        """
        # Classification layers
        output = self.dropout(self.fc(src))
        # output = F.log_softmax(self.dropout(self.fc(x)), dim=1) # Not necessary if using CrossEntropyLoss
        return output


#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#

class Experiment(object):
    def __init__(self, parameters_exp):
        """
            Class that trains a model with multple input features

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

        # Model type to use
        if ('model_type' not in parameters_exp):
            parameters_exp['model_type'] = 'Hybrid'
        self.model_type = parameters_exp['model_type']

        # Precise model to use
        if ('model_to_use' not in parameters_exp):
            if (parameters_exp['model_type'].lower() == 'BimodalTransformerCNN'):
                parameters_exp['model_to_use'] = 'BimodalCNNTransformer_Raw+Spec'
            else:
                raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))
        self.model_to_use = parameters_exp['model_to_use']

        # Some Transformer parameters
        if ('transformer' in self.model_to_use.lower()):
            # Some parameters needed to create Transformer models
            tmp_d_models = []
            for parameter_name in parameters_exp:
                if ('d_model' in parameter_name):
                    tmp_d_models.append(parameters_exp[parameter_name])
            self.d_model_max = max(tmp_d_models)
            self.d_model_min = min(tmp_d_models)
            if ('context_length' not in parameters_exp):
                parameters_exp['context_length'] = 1
            # Use class token parameter if the model type is a Transformer
            if (parameters_exp['model_type'].lower() == 'transformer'):
                if ('classification_pool' not in parameters_exp):
                    parameters_exp['classification_pool'] = 'ClassToken'
            # Project the input data before feeding it to the Transformer
            if ('project_input' not in parameters_exp):
                parameters_exp['project_input'] = True
            else:
                if (type(parameters_exp['project_input']) != bool):
                    parameters_exp['project_input'] = string_to_bool(parameters_exp['project_input'])

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
        if ('use_iterated_loss' not in parameters_exp):
            parameters_exp['use_iterated_loss'] = False
        else:
            parameters_exp['use_iterated_loss'] = string_to_bool(parameters_exp['use_iterated_loss'])
        self.use_iterated_loss = parameters_exp['use_iterated_loss']
        if ('model_apply_iterated_loss' not in parameters_exp):
            parameters_exp['model_apply_iterated_loss'] = 'RawEncoder'
        self.model_apply_iterated_loss = parameters_exp['model_apply_iterated_loss']
        if (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
            # Importance of the raw encoder iterated loss
            if ('importance_iterated_loss_raw' not in parameters_exp):
                parameters_exp['importance_iterated_loss_raw'] = 1.0
            self.importance_iterated_loss_raw = parameters_exp['importance_iterated_loss_raw']
            # Importance of the spec encoder iterated loss
            if ('importance_iterated_loss_spec' not in parameters_exp):
                parameters_exp['importance_iterated_loss_spec'] = 1.0
            self.importance_iterated_loss_spec= parameters_exp['importance_iterated_loss_spec']
        else:
            if ('importance_iterated_loss' not in parameters_exp):
                parameters_exp['importance_iterated_loss'] = 1.0
            self.importance_iterated_loss = parameters_exp['importance_iterated_loss']

        # Warmup steps (for NOAM when it is used)
        if ('warmup' not in parameters_exp):
            parameters_exp['warmup'] = 4000
        self.warmup = parameters_exp['warmup']

        # Fusion strategy (if one can be given to the model)
        if ('fusion_strategy' not in parameters_exp):
            parameters_exp['fusion_strategy'] = 'sum'

        # Creation of val dataset ?
        if ('separate_val_ds' not in parameters_exp):
            parameters_exp['separate_val_ds'] = True
            # parameters_exp['separate_val_ds'] = False
        self.separate_val_ds = parameters_exp['separate_val_ds']

        # Compute class weights parameter
        if ('compute_class_weights' not in parameters_exp):
            parameters_exp['compute_class_weights'] = True
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
            self.parameters_exp = parameters_exp

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

        # Parameter for the regularization terms
        # DEC
        if ('use_DEC' not in parameters_exp):
            parameters_exp['use_DEC'] = False
        else:
            parameters_exp['use_DEC'] = string_to_bool(parameters_exp['use_DEC'])
        self.use_DEC = parameters_exp['use_DEC']
        if ('models_to_apply_DEC' not in parameters_exp):
            parameters_exp['models_to_apply_DEC'] = ['Encoder']
        self.models_to_apply_DEC = parameters_exp['models_to_apply_DEC']
        if (self.use_DEC):
            # Defining the parameters for the loss
            if ('alpha_dec' not in parameters_exp):
                parameters_exp['alpha_dec'] = 1.0
            self.alpha_dec = parameters_exp['alpha_dec']
            if ('importance_dec' not in parameters_exp):
                parameters_exp['importance_dec'] = 1.0
            self.importance_dec = {}
            if (type(parameters_exp['importance_dec']) == float):
                for model_to_apply_DEC in self.models_to_apply_DEC:
                    self.importance_dec[model_to_apply_DEC] = parameters_exp['importance_dec']
            elif (type(parameters_exp['importance_dec']) == dict):
                self.importance_dec = parameters_exp['importance_dec']
            else:
                raise TypeError("The parameter importance_dec has to be a float or a dict")
            parameters_exp['importance_dec'] = self.importance_dec
        # Defining the epoch from which we will use DEC
        if ('epoch_init_dec_loss' not in parameters_exp):
            parameters_exp['epoch_init_dec_loss'] = 2
        self.epoch_init_dec_loss = parameters_exp['epoch_init_dec_loss']

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
        test_sampler = SubsetRandomSampler(test_indices)
        # test_sampler = SequentialSampler(test_indices)
        self.test_loader = torch.utils.data.DataLoader(self.test_ds,\
                                                       batch_size=self.batch_size_test,\
                                                       sampler=test_sampler)

    def modelCreation(self):
        """
            Creates a model to be trained on the selected time-frequency
            representation
        """
        # Creating the model
        if (self.model_type.lower() == 'hybrid'):
            if (self.model_to_use.lower() == 'bimodalcnntransformer_raw+spec'):
                data_shape = self.audio_feature_shapes['Spectrogram']
                self.model = TransformerClassifierBimodal_RawAndSpec(
                                                                        self.parameters_exp['in_channels'],
                                                                        self.parameters_exp['nhead'],
                                                                        self.parameters_exp['d_hid'],
                                                                        self.parameters_exp['nlayers'],
                                                                        self.parameters_exp['dropout'],
                                                                        self.parameters_exp['nb_features_projection'],
                                                                        self.parameters_exp['d_model_raw'],
                                                                        self.nb_classes,
                                                                        self.parameters_exp['classification_pool'],
                                                                        self.parameters_exp['n_conv_layers'],
                                                                        self.parameters_exp['nb_init_filters'],
                                                                        self.parameters_exp['increase_nb_filters_mode'],
                                                                        self.parameters_exp['pooling_mode'],
                                                                        data_shape,
                                                                        self.parameters_exp['fusion_strategy'],
                                                                        self.parameters_exp['dim_common_space']
                                                                    )
            else:
                raise ValueError("Model to use {} is not valid".format(self.model_to_use))

        else:
            raise ValueError("Model type {} is not valid".format(self.model_type))


        # Sending the model to the correct device
        self.model.to(self.device)

        # Getting the hidden dimension of the encoded representation used for
        # classification (necessary to DEC loss)
        dummy_data = {}
        for feature_type in self.audio_feature_shapes:
            dummy_data[feature_type] = torch.randn([1] + [el for el in self.audio_feature_shapes[feature_type]]).to(self.device)
        self.hidden_dimension = {}
        if (self.model_to_use.lower() == 'bimodalcnntransformer_raw+spec'):
            # General encoder
            self.hidden_dimension['Encoder'] = self.model.encoder(dummy_data).shape[1]
            # Raw encoder
            self.hidden_dimension['Raw_Encoder'] = self.model.encoder.raw_encoder(dummy_data['RawSignal']).shape[1]
            # Spec Encoder
            self.hidden_dimension['Spec_Encoder'] = self.model.encoder.spec_encoder(dummy_data['Spectrogram']).shape[1]
        else:
            raise ValueError("Model to use {} is not valid".format(self.model_to_use))

        # # Summary of the model
        # print("\n\n Model summary")
        # summary(self.model, self.audio_feature_shapes)
        # print("\n\n")

        # Creating model for iterated loss
        if (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
            # Raw encoder iterated loss
            self.raw_model_iterated_loss_classifier = IteratedLossClassifierRawSignal(self.parameters_exp['d_model_raw'], self.nb_classes)
            self.raw_model_iterated_loss_classifier.to(self.device)
            # Spec encoder iterated loss
            self.spec_model_iterated_loss_classifier = IteratedLossClassifierSpectrogram(self.hidden_dimension['Spec_Encoder'], self.nb_classes, dropout_probability=self.parameters_exp['dropout'])
            self.spec_model_iterated_loss_classifier.to(self.device)
        else:
            if (self.model_apply_iterated_loss.lower() == 'rawencoder'):
                self.model_iterated_loss_classifier = IteratedLossClassifierRawSignal(self.parameters_exp['d_model_raw'], self.nb_classes)
            elif (self.model_apply_iterated_loss.lower() == 'specencoder'):
                self.model_iterated_loss_classifier = IteratedLossClassifierSpectrogram(self.hidden_dimension['Spec_Encoder'], self.nb_classes, dropout_probability=self.parameters_exp['dropout'])
            self.model_iterated_loss_classifier.to(self.device)


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
        else:
            raise ValueError('Loss function {} is not valid'.format(self.parameters_exp['loss_function']))

    def initialize_dec_loss(self):
        """
            Code strongly inspired from
            https://github.com/vlukiyanov/pt-dec/blob/master/ptdec/model.py
            lines 74-97
        """
        # Putting the right hidden dimension in the attributes of the DEC loss
        self.dec_loss = {}
        for model_to_apply_DEC in self.models_to_apply_DEC:
            self.dec_loss[model_to_apply_DEC] = DECLoss(
                                                            cluster_number=self.nb_classes,
                                                            hidden_dimension=self.hidden_dimension[model_to_apply_DEC], # Temporary, it will be updated later, during the initialization of DEC
                                                            alpha=self.alpha_dec,
                                                        ).to(self.device)

        # Creating an instance of the k-means method that is going to be used
        # to create the inital centroids
        kmeans = KMeans(n_clusters=self.nb_classes, n_init=20)

        # Computing the centroids using the embedded representations learned
        # by the model
        self.model.train()
        features = {}
        for batch in self.train_loader:
            audio_features, labels = batch
            audio_features, labels = {feature_type: audio_features[feature_type].float().to(self.device) for feature_type in audio_features}, labels.to(self.device)
            for model_to_apply_DEC in self.models_to_apply_DEC:
                # Creating the list of features for the desired model
                if (model_to_apply_DEC not in features):
                    features[model_to_apply_DEC] = []
                # Filling the list of features
                if (model_to_apply_DEC.lower() == 'encoder'):
                    features[model_to_apply_DEC].append(self.model.encoder(audio_features).detach().cpu())
                if (model_to_apply_DEC.lower() == 'raw_encoder'):
                    features[model_to_apply_DEC].append(self.model.encoder.raw_encoder(audio_features['RawSignal']).detach().cpu())
                if (model_to_apply_DEC.lower() == 'spec_encoder'):
                    features[model_to_apply_DEC].append(self.model.encoder.spec_encoder(audio_features['Spectrogram']).detach().cpu())

        # Predicted clusters and assigning the clusters centers to each DEC loss
        predicted = {}
        cluster_centers = {}
        for model_to_apply_DEC in self.models_to_apply_DEC:
            predicted[model_to_apply_DEC] = kmeans.fit_predict(torch.cat(features[model_to_apply_DEC]).numpy())
            cluster_centers[model_to_apply_DEC] = torch.tensor(
                                                                    kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
                                                              ).to(self.device)
            with torch.no_grad():
                # initialise the cluster centers
                self.dec_loss[model_to_apply_DEC].assignment.cluster_centers.copy_(cluster_centers[model_to_apply_DEC])
                # self.dec_loss[model_to_apply_DEC].state_dict()["assignment.cluster_centers"].copy_(cluster_centers[model_to_apply_DEC])

        # Adding the cluster centers to the learnable parameters of the optimizer
        if (self.use_iterated_loss):
            if (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
                self.createOptimizer(list(self.model.parameters()) + list(self.raw_model_iterated_loss_classifier.parameters()) + list(self.spec_model_iterated_loss_classifier.parameters()) + [self.dec_loss[model_to_apply_DEC].assignment.cluster_centers for model_to_apply_DEC in self.models_to_apply_DEC])
            else:
                self.createOptimizer(list(self.model.parameters()) + list(self.model_iterated_loss_classifier.parameters()) + [self.dec_loss[model_to_apply_DEC].assignment.cluster_centers for model_to_apply_DEC in self.models_to_apply_DEC])
        else:
            self.createOptimizer(list(self.model.parameters()) + [self.dec_loss[model_to_apply_DEC].assignment.cluster_centers for model_to_apply_DEC in self.models_to_apply_DEC])


    def createOptimizer(self, model_parameters):
        # Creating the optimizer
        self.optimizer = noam.Noam(
                                    params=model_parameters,
                                    lr=self.lr,
                                    betas=(0.9, 0.999),
                                    eps=1e-16,
                                    optimized_update=True,
                                    amsgrad=False,
                                    warmup=self.warmup,
                                    d_model=self.d_model_max,
                                    # d_model=self.d_model_min,
                                )


    def compute_forward_pass(self, batch, epoch_nb, keep_grad=True):
        # Getting the data and the labels
        audio_features, labels = batch
        labels = labels.to(self.device)
        tmp_audio_features = {}
        for feature_type in audio_features:
            tmp_audio_features[feature_type] = audio_features[feature_type].float().to(self.device)
        audio_features = tmp_audio_features

        # Computing the loss
        # The DEC term HAS TO BE COMPUTED BEFORE the forward pass
        if (keep_grad) and (self.use_DEC) and (epoch_nb >= self.epoch_init_dec_loss):
            dec_loss = 0
            for model_to_apply_DEC in self.models_to_apply_DEC:
                if (model_to_apply_DEC.lower() == 'encoder'):
                    dec_loss += self.importance_dec[model_to_apply_DEC]*self.dec_loss[model_to_apply_DEC](self.model.encoder, audio_features)
                if (model_to_apply_DEC.lower() == 'raw_encoder'):
                    dec_loss += self.importance_dec[model_to_apply_DEC]*self.dec_loss[model_to_apply_DEC](self.model.encoder.raw_encoder, audio_features['RawSignal'])
                if (model_to_apply_DEC.lower() == 'spec_encoder'):
                    dec_loss += self.importance_dec[model_to_apply_DEC]*self.dec_loss[model_to_apply_DEC](self.model.encoder.spec_encoder, audio_features['Spectrogram'])

        # Iterated loss
        if (self.use_iterated_loss):
            if (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
                out_iterated_loss_raw_encoder = 0
                out_iterated_loss_spec_encoder  = 0
            else:
                out_iterated_loss = 0
            if (keep_grad):
                # Raw encoder only
                if (self.model_apply_iterated_loss.lower() == 'rawencoder'):
                    out_iterated_loss += self.model_iterated_loss_classifier(self.model.encoder.raw_encoder(audio_features['RawSignal']))
                # Spectrogram encoder only
                elif (self.model_apply_iterated_loss.lower() == 'specencoder'):
                    out_iterated_loss += self.model_iterated_loss_classifier(self.model.encoder.spec_encoder(audio_features['Spectrogram']))
                # Raw and Spectrogram encoders
                elif (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
                    # Treating the raw encoder
                    out_iterated_loss_raw_encoder += self.raw_model_iterated_loss_classifier(self.model.encoder.raw_encoder(audio_features['RawSignal']))
                    # Treating the spectrogram encoder
                    if (self.model_to_use.lower() == 'bimodalcnntransformer_raw+spec'):
                        out_iterated_loss_spec_encoder += self.spec_model_iterated_loss_classifier(self.model.encoder.spec_encoder(audio_features['Spectrogram']))
                # Getting the total iterated loss
                if (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
                    iterated_loss = self.importance_iterated_loss_raw * self.criterion(out_iterated_loss_raw_encoder, labels.long()) + self.importance_iterated_loss_spec * self.criterion(out_iterated_loss_spec_encoder, labels.long())
                else:
                    iterated_loss = self.criterion(out_iterated_loss, labels.long())
            else:
                with torch.no_grad():
                    # Raw encoder only
                    if (self.model_apply_iterated_loss.lower() == 'rawencoder'):
                        out_iterated_loss += self.model_iterated_loss_classifier(self.model.encoder.raw_encoder(audio_features['RawSignal']))
                    # Spectrogram encoder only
                    elif (self.model_apply_iterated_loss.lower() == 'specencoder'):
                        out_iterated_loss += self.model_iterated_loss_classifier(self.model.encoder.spec_encoder(audio_features['Spectrogram']))
                    # Raw and Spectrogram encoders
                    elif (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
                        # Treating the raw encoder
                        out_iterated_loss_raw_encoder += self.raw_model_iterated_loss_classifier(self.model.encoder.raw_encoder(audio_features['RawSignal']))
                        # Treating the spectrogram encoder
                        out_iterated_loss_spec_encoder += self.spec_model_iterated_loss_classifier(self.model.encoder.spec_encoder(audio_features['Spectrogram']))
                    # Getting the total iterated loss
                    if (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
                        iterated_loss = self.importance_iterated_loss_raw * self.criterion(out_iterated_loss_raw_encoder, labels.long()) + self.importance_iterated_loss_spec * self.criterion(out_iterated_loss_spec_encoder, labels.long())
                    else:
                        iterated_loss = self.criterion(out_iterated_loss, labels.long())

        # Computing the classification loss
        out = self.model(audio_features) # Generate predictions
        classification_loss = self.criterion(out, labels.long()) # Calculate loss

        # Final loss
        if (self.use_iterated_loss):
            if (keep_grad):
                if (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
                    loss = classification_loss + iterated_loss # ATTENTION: The hyper-parameters of the importance of the iterated loss of each modality are applied upper in this code
                else:
                    loss = classification_loss + self.importance_iterated_loss * iterated_loss
            else:
                loss = classification_loss
        else:
            loss = classification_loss

        if (keep_grad) and (self.use_DEC) and (epoch_nb >= self.epoch_init_dec_loss):
            # print("===> Using DEC Loss: {} ({})".format(dec_loss, type(dec_loss)))
            # print("\tLoss before adding DEC term: ", loss, dec_loss.requires_grad)
            loss += dec_loss
            # print("\tLoss after adding DEC term: ", loss)

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

    def single_train(self, save_checkpoints=True):
        """
            Trains a model one time during self.nb_epochs epochs
        """
        # Creating the dataloaders
        self.dataloadersCreation()

        # Creating the model
        self.modelCreation()

        # Creating the optimizer
        if (self.use_iterated_loss):
            if (self.model_apply_iterated_loss.lower() == 'rawencoder+specencoder'):
                self.createOptimizer( list(self.model.parameters()) + list(self.raw_model_iterated_loss_classifier.parameters()) + list(self.spec_model_iterated_loss_classifier.parameters()) )
            else:
                self.createOptimizer( list(self.model.parameters()) + list(self.model_iterated_loss_classifier.parameters()) )
        else:
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
            # Initializing the DEC Loss if used
            if (epoch == self.epoch_init_dec_loss):
                if (self.use_DEC):
                    # This HAS TO BE DONE BEFORE THE FORWARD PASS as new parameters
                    # are added to the optimizer (and, if we implemented it, the
                    # the optimizer can be restarted !)
                    self.initialize_dec_loss()

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
                    # if (self.model_type.lower() != 'transformer'):
                    if (self.model_type.lower() != 'transformer')\
                        and (self.model_type.lower() == 'hybrid' and 'transformer' not in self.model_to_use.lower()):
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
                print("\n=======>Train F1 Score at epoch {} is {}\n".format(epoch, f1_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch], average='micro')))
                if (self.separate_val_ds):
                    print("\t\tVal F1 Score at epoch {} is {}".format(epoch, f1_score(predictions_results['Val']['TrueLabels'][epoch], predictions_results['Val']['PredictedLabels'][epoch], average='macro')))
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

            if (save_checkpoints):
                # Saving the model and the current results
                current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
                # Model
                torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': train_loss
                            }, self.results_folder + '/model/checkpoint_model-{}.pth'.format(self.exp_id))
                # Current results
                checkpoint_results_file = self.results_folder + '/metrics/checkpoint_results-{}_epoch-{}_{}.pth'.format(self.exp_id, epoch, current_datetime)
                intermediate_results = {'Loss': loss_values, 'Predictions': predictions_results}
                with open(checkpoint_results_file, "wb") as fp:   #Pickling
                    pickle.dump(intermediate_results, fp)

        return {'Loss': loss_values, 'Predictions': predictions_results}


    def holdout_train(self):
        """
            Does a holdout training repeated self.nb_repetitions times
        """
        repetitionsResults = {}
        for nb_repetition in range(self.nb_repetitions):
            print("\n\n=======> Repetitions {} <=======".format(nb_repetition))
            # Doing single train
            tmp_results = self.single_train(save_checkpoints=True)
            repetitionsResults[nb_repetition] = tmp_results

            # Saving the final model and the results
            # Model
            torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'model': self.model
                        }, self.results_folder + '/model/final_model-{}_rep-{}.pth'.format(self.exp_id, nb_repetition))
            # Results
            with open(self.results_folder + '/metrics/results_exp-{}_rep-{}.pth'.format(self.exp_id, nb_repetition), "wb") as fp:   #Pickling
                pickle.dump(tmp_results, fp)

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
        download_results_experiment(experiment_name='Exp-1_MultiFeature', local_data_directory='../../../results/')
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
        if (exp.dataset_type.lower() == 'eegepilepticseizure'):
            shutil.copy2(parameters_exp['dataset_folder'] + '/data.hdf5', resultsFolder + '/params_exp/data.hdf5')


    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()
