#!/usr/bin/env python3
"""
    Code for the ECG Heartbeat Categorization Datasets
"""
import csv
import numpy as np

import random
from random import shuffle
from random import randint

import torch
from torch.utils.data import Dataset

from scipy import signal

import matplotlib.pyplot as plt

import src.pycochleagram.cochleagram as cgram
from src.utils.raw_audio_tools import get_feature
from src.utils.spectrogram import stft_signal, estimate_inf_sup_db


def open_csv_ecg_heartbeat_categorization_datasets(csv_file):
    """
        Opens a csv file from one of the two ECG Heartbeat Categorization
        Datasets.

        Arguments:
        ----------
        csv_file: str
            Path to a csv file from one of the ECG Heartbeat Categorization
            datasets.

        Returns:
        --------
        data: dict
            Dictionary containing the data in the csv file. Each key correspond
            to the identifier of the sample and the value is another dict with
            two keys: "Data" (ECG signal) and 'Label'.
    """
    # Loading the samples
    tmp_samples = []
    with open(csv_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            ecg_data = np.array(row[:-1])
            label = row[-1]
            sample = {
                        'Data': ecg_data,
                        'Label': label
                     }
            tmp_samples.append(sample)

    # Mixing the samples
    random.Random(42).shuffle(tmp_samples)

    # Creating the data dict
    data = {}
    nb_samples_per_class = {}
    for sample_id in range(len(tmp_samples)):
        # Saving the sample
        sample = tmp_samples[sample_id]
        data[sample_id] = sample

        # Counting the number of samples
        if (sample['Label'] not in nb_samples_per_class):
            nb_samples_per_class[sample['Label']] = 1
        else:
            nb_samples_per_class[sample['Label']] += 1

    return data


def load_ecg_datasets_heartbeat_categorization(dataset_folder):
    """
        Load the dataset "ECG Heartbeat Categorization Dataset" from
        the Kaggle challenge https://www.kaggle.com/shayanfazeli/heartbeat.
        This dataset is in fact composed of two datasets: the Arrhythmia Dataset
        (5 classes) and the PTB Diagnostic ECG Database (2 classes)

        Arguments:
        ----------
        dataset_folder: str
            Path to the folder containing the data of the dataset.

        Returns:
        --------
        datasets: dict
            Dictionary with two keys: 'MITBIH' and 'PTB'. The values are also
            dictionaries with two keys, 'Train' and 'Test', corresponding to the
            training and testing splits.
    """
    #=============================================================================#
    #=============================================================================#
    # Arrhythmia Dataset
    # Loading the data of the Arrhythmia Dataset
    train_data_mitbih = open_csv_ecg_heartbeat_categorization_datasets(dataset_folder + '/mitbih_train.csv')
    test_data_mitbih = open_csv_ecg_heartbeat_categorization_datasets(dataset_folder + '/mitbih_test.csv')

    #=============================================================================#
    #=============================================================================#
    # PTB Diagnostic ECG Dataset
    # Loading the data of the PTB Diagnostic ECG Dataset
    normal_data_ptb = open_csv_ecg_heartbeat_categorization_datasets(dataset_folder + '/ptbdb_normal.csv')
    abnormal_data_ptb = open_csv_ecg_heartbeat_categorization_datasets(dataset_folder + '/ptbdb_abnormal.csv')
    # Computing the number of samples per set split
    nb_normal_ptb, nb_abnormal_ptb = len(normal_data_ptb), len(abnormal_data_ptb)
    percentage_train = 0.8
    nb_normal_train_samples, nb_abnormal_train_samples = int(round(percentage_train*nb_normal_ptb)), int(round(percentage_train*nb_abnormal_ptb))
    normal_ptb_idxs, abnormal_ptb_idxs = list(normal_data_ptb.keys()), list(abnormal_data_ptb.keys())
    # Training data
    tmp_train_samples = [normal_data_ptb[sample_id] for sample_id in normal_ptb_idxs[0:nb_normal_train_samples]]
    tmp_train_samples += [abnormal_data_ptb[sample_id] for sample_id in abnormal_ptb_idxs[0:nb_abnormal_train_samples]]
    shuffle(tmp_train_samples)
    train_data_ptb = {}
    for sample_id in range(len(tmp_train_samples)):
        train_data_ptb[sample_id] = tmp_train_samples[sample_id]
    # Test data
    tmp_test_samples = [normal_data_ptb[sample_id] for sample_id in normal_ptb_idxs[nb_normal_train_samples:]]
    tmp_test_samples += [abnormal_data_ptb[sample_id] for sample_id in abnormal_ptb_idxs[nb_abnormal_train_samples:]]
    shuffle(tmp_test_samples)
    test_data_ptb = {}
    for sample_id in range(len(tmp_test_samples)):
        test_data_ptb[sample_id] = tmp_test_samples[sample_id]

    #=============================================================================#
    #=============================================================================#
    # Creating the final datasets
    datasets = {
                    'MITBIH': {'Train': train_data_mitbih, 'Test': test_data_mitbih},
                    'PTB': {'Train': train_data_ptb, 'Test': test_data_ptb}
               }
    return datasets


# Creating the new dataste class
class EcgCategorization(Dataset):
    """
        data: dict
        features_types: str
            Features to use. Different options are possible:
                - RawSignal
                - Spectrogram
        add_channel_dim: bool
            True if we want to add a dimension for the channel. This is useful
            when using 2D convolutions.
        params: dict
            Parameters to compute the selected representation. This parameters
            vary from one representation to another
        means: dict
            Dictionary of Means to center the samples of the dataset. The keys
            are the features that are going to be used
        stds: dict
            Dictionary of Stds to reduce the samples of the dataset. The keys
            are the features that are going to be used
    """
    def __init__(self, data, feature_type, add_channel_dim=False, params={}, mean=0, std=1):
        super().__init__()
        self.data = data
        self.sample_rate = 125 # Hz
        self.feature_type = feature_type
        self.add_channel_dim = add_channel_dim
        self.params = params
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Getting the raw data
        signal, label = self.data[i]['Data'], self.data[i]['Label']

        # Getting the desired feature
        feature = get_feature(
                                        signal,
                                        self.sample_rate,
                                        self.add_channel_dim,
                                        self.feature_type,
                                        self.params
                                    )


        return torch.from_numpy(feature.copy()).float(), label


# Creating the new dataste class
class EcgCategorization_Multifeature(Dataset):
    """
        data: dict
        features_types: list (of str)
            List of (str) features to use
            Different options are possible:
                - RawSignal
                - Spectrogram
        add_channel_dim: bool
            True if we want to add a dimension for the channel. This is useful
            when using 2D convolutions.
        params: dict
            Parameters to compute the selected representation. This parameters
            vary from one representation to another
        means: dict
            Dictionary of Means to center the samples of the dataset. The keys
            are the features that are going to be used
        stds: dict
            Dictionary of Stds to reduce the samples of the dataset. The keys
            are the features that are going to be used
    """
    def __init__(self, data, features_types, add_channel_dim=True, params={}, means=[], stds=[]):
        super().__init__()
        self.data = data
        self.sample_rate = 125 # Hz
        self.features_types = features_types
        self.add_channel_dim = add_channel_dim
        self.params = params
        if (len(means) == 0) or (len(stds) == 0):
            self.means, self.stds = {}, {}
            for feature_type in self.features_types:
                self.means[feature_type] = 0
                self.stds[feature_type] = 1
        else:
            self.means = means
            self.stds = stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Getting the raw data
        signal, label = self.data[i]['Data'], self.data[i]['Label']

        # Getting the desired feature
        features = {}
        for feature_type in self.features_types:
            feature = get_feature(
                                                signal,
                                                self.sample_rate,
                                                self.add_channel_dim,
                                                feature_type,
                                                self.params
                                            )
            feature = (feature - self.means[feature_type])/self.stds[feature_type]
            torch.from_numpy(feature.copy()).float()
            features[feature_type] = feature


        return features, label

#===============================================================================#
#===============================================================================#
def main():
    # Getting the datasets
    datasets = load_ecg_datasets_heartbeat_categorization('../../data/ECG_Heartbeat_Categorization_Dataset/')

    # Plotting a ECG signal
    sample_rate = 125
    # dataset_name = 'MITBIH'
    dataset_name = 'PTB'
    sample_id = randint(0, len(datasets[dataset_name]['Train']))
    sample = datasets[dataset_name]['Train'][sample_id]
    sample_data, sample_label = sample['Data'], sample['Label']
    x = [i for i in range(len(sample_data))]
    y = sample_data
    plt.plot(x, y, label=sample_label)
    plt.legend()
    plt.show()

    # Creating some representations
    # Spectrogram
    # n_fft, hop_length, window = 64, 16, 'blackman'
    n_fft, hop_length, window = 32, 4, 'blackman'
    stft_waveform, f, t = stft_signal(sample_data, n_fft, sample_rate, hop_length, window)
    spectrogram = np.absolute(stft_waveform)**2
    log_spec = 20*np.log10(spectrogram)
    # Estimating the min and max db values of the spectrogram in log scale for the colobar
    x, y = 0, 9
    min_db, max_db = estimate_inf_sup_db(spectrogram, x, y)
    print("Min_db = {}, Max_db = {}".format(min_db, max_db))
    # Filtering the spectrogram
    log_spec = 20*np.log10(spectrogram)
    log_spec[log_spec < min_db] = min_db
    log_spec[log_spec > max_db] = max_db
    # Plotting
    plt.pcolormesh(log_spec, shading='gouraud', cmap='viridis')
    plt.title("Spectrogram")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


if __name__=='__main__':
    main()
