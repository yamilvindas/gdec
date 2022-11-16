#!/usr/bin/env python3
"""
    This code allows to take a trained model and plot into a 2D space the
    learned representations

    Options:
    --------
    --parameters_file: str
        Path to a file containing the parameters of the experiment.
        This files are usually located in /hits_signal_learning/parameters_files/sub-folders/
    --model_params_file: str
        Parameters of the original experment allowing to obtain the trained model (pth file)
    --model_weights_file: str
        Weights of the trained model (pth file)
    --encoder_to_use: str
        Name of the encoder to use. Four options: Whole, Raw_Encoder, Spec_Encoder and Time_Freq_Encoder
"""
import json
import pickle
import argparse

import matplotlib.pyplot as plt

import random

import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from sklearn.manifold import TSNE
from umap import UMAP

from src.utils.tools import string_to_bool, train_val_split_stratified
from src.DataManipulation.ecg_data import EcgCategorization_Multifeature, load_ecg_datasets_heartbeat_categorization
from src.DataManipulation.eeg_data import EEG_EpilepticSeizureRecognition_Multifeature, loadFromHDF5_EEG

# Models
from src.Models.Hybrid.Transformer_CNN_RawAndSpec import TransformerClassifierBimodal_RawAndSpec

#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#

class Experiment(object):
    def __init__(self, parameters_exp):
        """
            Plot of the 2D projections of the embeddings obtained with a the
            encoder of a trained multi-feature model

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

        # Defining the weights of the model
        if ('model_weights_file' not in parameters_exp):
            raise ValueError('The user should specify the weights file of the model to use!')
        self.model_weights_file = parameters_exp['model_weights_file']

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
            if ('d_model' not in parameters_exp):
                parameters_exp['d_model'] = 64
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

        # Encoder to use
        self.encoder_to_use = parameters_exp['encoder_to_use']

        # Parameters for data loading
        self.batch_size_train = parameters_exp['batch_size_train']
        self.batch_size_test = parameters_exp['batch_size_test']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Creation of val dataset ?
        if ('separate_val_ds' not in parameters_exp):
            parameters_exp['separate_val_ds'] = True
            # parameters_exp['separate_val_ds'] = False
        self.separate_val_ds = parameters_exp['separate_val_ds']

        # Dataset loading
        if (self.dataset_type.lower()=='ecgcategorization'):
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
        print("Number of samples in the testing dataset: ", len(self.test_ds))
        if (self.separate_val_ds):
            print("Number of samples in the validation dataset: ", len(self.val_ds))

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

    def load_weights_model(self):
        """
            Loads the weights of a trained model.
        """
        # Loading the data of a model
        model_data = torch.load(self.model_weights_file, map_location=torch.device('cpu'))

        # Loading the weights into the model
        self.model.load_state_dict(model_data['model_state_dict'])
        print("===> Model loaded successfully !")

    def compute_encoding(self, batch):
        """
        Gets the encoding of a (set of) sample(s) using the encoder of
        the model.
        """
        # Getting the data and the labels
        audio_features, labels = batch
        audio_features, labels = {feature_type: audio_features[feature_type].float().to(self.device) for feature_type in audio_features}, labels.to(self.device)

        # Computing the encoding
        if (self.encoder_to_use.lower() == 'whole'):
            encoding = self.model.encoder(audio_features) # Generate predictions
        elif (self.encoder_to_use.lower() == 'raw_encoder'):
            encoding = self.model.encoder.raw_encoder(audio_features['RawSignal']) # Generate predictions
        elif (self.encoder_to_use.lower() == 'spec_encoder'):
            encoding = self.model.encoder.spec_encoder(audio_features['Spectrogram']) # Generate predictions

        # Getting the predictions
        samples_embeddings = []
        for i in range(len(encoding)):
            true_class = int(labels[i])
            embedding = encoding[i].detach().cpu().numpy()
            sample_to_append = {
                                    "Embedding": embedding,
                                    "Label": int(true_class),
                                    "xCoord": None,
                                    "yCoord": None
                                }
            samples_embeddings.append(sample_to_append)

        return samples_embeddings


    def single_pass(self):
        """
            Iterates over all the samples in the training, validation and
            test sets to compute the embeddings
        """
        # TODO
        # Creating the dataloaders
        self.dataloadersCreation()

        # Creating the model
        self.modelCreation()

        # Loading the weights
        self.load_weights_model()

        # Data structures for the losses and the predictions
        embeddings = {
                        'Train': [],
                        'Val': [],
                        'Test': []
                      }

        # Computing the embeddings
        self.model.eval()
        # Training data
        with (torch.no_grad()):
            for batch in self.train_loader:
                # Embeddings of the batch
                train_samples_embeddings = self.compute_encoding(batch)
                embeddings['Train'] += train_samples_embeddings

        # Validation data
        if (self.separate_val_ds):
            with (torch.no_grad()):
                for batch in self.val_loader:
                    # Embeddings of the batch
                    val_samples_embeddings = self.compute_encoding(batch)
                    embeddings['Val'] += val_samples_embeddings

        # Test data
        with (torch.no_grad()):
            for batch in self.test_loader:
                # Embeddings of the batch
                test_samples_embeddings = self.compute_encoding(batch)
                embeddings['Test'] += test_samples_embeddings

        return embeddings

    def set_proj_save_folder(self, save_proj_folder):
        """
            Sets the folder where the projections are going to be stored
        """
        self.save_proj_folder = save_proj_folder

    def projectEmbeddings(self):
        """
            Projects the embeddings (extracted with compute_encoding) of the
            samples in a 2D space.
        """
        # Computing the embeddings
        embeddings = self.single_pass()
        print("===> Embedded representations computed !")

        # Reshaping the embeddings to be able to project them
        if (len(embeddings['Train'][0]['Embedding'].shape) != 1):
            raise ValueError('Shape {} is not compatible with t-SNE of Scikit Learn'.format(embeddings['Train'][0]['Embedding'].shape))

        # Getting the list of samples to project
        # Train
        embedded_representations_train = []
        labels_train = []
        for sample in embeddings['Train']:
            embedded_representations_train.append(sample['Embedding'])
            labels_train.append(sample['Label'])
        # Val
        embedded_representations_val = []
        labels_val = []
        for sample in embeddings['Val']:
            embedded_representations_val.append(sample['Embedding'])
            labels_val.append(sample['Label'])
        # Test
        embedded_representations_test = []
        labels_test = []
        for sample in embeddings['Test']:
            embedded_representations_test.append(sample['Embedding'])
            labels_test.append(sample['Label'])


        # Projecting the samples in a 2D space
        # Train
        # final_train_representations = TSNE(n_components=2).fit_transform(embedded_representations_train)
        final_train_representations = UMAP(n_components=2, n_neighbors=5).fit_transform(embedded_representations_train)
        SC_train_representations = silhouette_score(final_train_representations, labels_train, metric='sqeuclidean')
        print("\n\n=======>Train 2D projections Silhouette Score: {}".format(SC_train_representations))
        # Val
        if (self.separate_val_ds):
            # final_val_representations = TSNE(n_components=2).fit_transform(embedded_representations_val)
            final_val_representations = UMAP(n_components=2, n_neighbors=5).fit_transform(embedded_representations_val)
            SC_val_representations = silhouette_score(final_val_representations, labels_val, metric='sqeuclidean')
            print("\n\n=======>Val 2D projections Silhouette Score: {}".format(SC_val_representations))
        # Test
        # final_test_representations = TSNE(n_components=2).fit_transform(embedded_representations_test)
        final_test_representations = UMAP(n_components=2, n_neighbors=5).fit_transform(embedded_representations_test)
        SC_test_representations = silhouette_score(final_test_representations, labels_test, metric='sqeuclidean')
        print("\n\n=======>Test 2D projections Silhouette Score: {}".format(SC_test_representations))
        print("===> t-SNE projection done !")

        # Creating the CSV files of each subset
        # Train
        for sample_id in range(len(embeddings['Train'])):
            _ = embeddings['Train'][sample_id].pop('Embedding')
            embeddings['Train'][sample_id]['xCoord'] = final_train_representations[sample_id, 0]
            embeddings['Train'][sample_id]['yCoord'] = final_train_representations[sample_id, 1]
        df_samples_train = pd.DataFrame(embeddings['Train'])
        df_samples_train.to_csv(self.save_proj_folder+'/tmp_train_UMAP_{}.csv'.format(self.encoder_to_use), index=False)
        # Val
        if (self.separate_val_ds):
            for sample_id in range(len(embeddings['Val'])):
                _ = embeddings['Val'][sample_id].pop('Embedding')
                embeddings['Val'][sample_id]['xCoord'] = final_val_representations[sample_id, 0]
                embeddings['Val'][sample_id]['yCoord'] = final_val_representations[sample_id, 1]
            df_samples_val = pd.DataFrame(embeddings['Val'])
            df_samples_val.to_csv(self.save_proj_folder+'/tmp_val_UMAP_{}.csv'.format(self.encoder_to_use), index=False)
        # Test
        for sample_id in range(len(embeddings['Test'])):
            _ = embeddings['Test'][sample_id].pop('Embedding')
            embeddings['Test'][sample_id]['xCoord'] = final_test_representations[sample_id, 0]
            embeddings['Test'][sample_id]['yCoord'] = final_test_representations[sample_id, 1]
        df_samples_test = pd.DataFrame(embeddings['Test'])
        df_samples_test.to_csv(self.save_proj_folder+'/tmp_test_UMAP_{}.csv'.format(self.encoder_to_use), index=False)

        # Plot the embedded representations
        # Train
        fig, ax = plt.subplots()
        scatter = ax.scatter(x=final_train_representations[:, 0], y=final_train_representations[:, 1], c=labels_train, s=5)
        legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        ax.add_artist(legend)
        plt.savefig(self.save_proj_folder+'/tmp_train_UMAP_{}.png'.format(self.encoder_to_use), dpi=300)

        # Validation
        if (self.separate_val_ds):
            fig, ax = plt.subplots()
            scatter = ax.scatter(x=final_val_representations[:, 0], y=final_val_representations[:, 1], c=labels_val, s=5)
            legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
            ax.add_artist(legend)
            plt.savefig(self.save_proj_folder+'/tmp_val_UMAP_{}.png'.format(self.encoder_to_use), dpi=300)

        # Test
        fig, ax = plt.subplots()
        scatter = ax.scatter(x=final_test_representations[:, 0], y=final_test_representations[:, 1], c=labels_test, s=5)
        legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        ax.add_artist(legend)
        plt.savefig(self.save_proj_folder+'/tmp_test_UMAP_{}.png'.format(self.encoder_to_use), dpi=300)

        # ALL
        final_all_representations = np.concatenate((final_train_representations, final_val_representations, final_test_representations), axis=0)
        labels_all = labels_train + labels_val + labels_test
        fig, ax = plt.subplots()
        scatter = ax.scatter(x=final_all_representations[:, 0], y=final_all_representations[:, 1], c=labels_all, s=5)
        legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        ax.add_artist(legend)
        plt.savefig(self.save_proj_folder+'/tmp_all_UMAP_{}.png'.format(self.encoder_to_use), dpi=300)
        print("Figures of the projections saved at: {}".format(self.save_proj_folder))



#==============================================================================#
#================================ Main Function ================================#
#==============================================================================#
def main():
    print("\n\n==================== Beginning of the experiment ====================\n\n")
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--parameters_file', help="Parameters for the experiment", type=str)
    ap.add_argument('--model_params_file', help="Parameters of the original experment allowing to obtain the trained model (pth file)", type=str)
    ap.add_argument('--model_weights_file', help="Weights of the trained model (pth file)", type=str)
    ap.add_argument('--encoder_to_use', default='Whole', help="Name of the encoder to use. Four options: Whole, Raw_Encoder, Spec_Encoder and Time_Freq_Encoder", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    if (args['parameters_file'] is None):
        if (args['model_params_file'] is None) or (args['model_weights_file'] is None):
            raise ValueError("If --parameters_file option is not used, --model_params_file AND --model_weights_file should be specified")
        else:
            model_params_file = args['model_params_file']
            model_weights_file = args['model_weights_file']
    else:
        parameters_file = args['parameters_file']
        print("=======>WARNING: Using the parameters in parameters file {}".format(parameters_file))
        with open(parameters_file) as jf:
            parameters_exp = json.load(jf)
        model_params_file = parameters_exp['model_params_file']
        model_weights_file = parameters_exp['model_weights_file']
    # Encoder to use
    encoder_to_use = args['encoder_to_use']

    # Loading the parameters necessary to build the model
    params_exp = None
    with open(model_params_file, 'rb') as pf:
        params_exp = pickle.load(pf)
    params_exp['model_weights_file'] = model_weights_file
    params_exp['encoder_to_use'] = encoder_to_use

    #==========================================================================#
    # Creating an instance of the experiment
    seed = 42
    random.seed(seed) # For reproducibility purposes
    np.random.seed(seed) # For reproducibility purposes
    torch.manual_seed(seed) # For reproducibility purposes
    if torch.cuda.is_available(): # For reproducibility purposes
        torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True) # For reproducibility purposes
    exp = Experiment(params_exp)

    # Defining the folder to save the projections
    proj_save_folder = '/'.join(model_params_file.split('/')[:-2]) + '/'
    exp.set_proj_save_folder(proj_save_folder)

    # Getting the projection of the embeddings
    exp.projectEmbeddings()


    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()
