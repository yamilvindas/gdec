#!/usr/bin/env python3
"""
    Code for the EEG Epileptic Recognition Dataset
"""
import csv
import argparse
import numpy as np

from random import shuffle
from random import randint

import torch
from torch.utils.data import Dataset

import h5py

from src.utils.raw_audio_tools import get_feature


# Loading the data
def open_csv_eeg_epileptic_recognition_dataset(csv_file):
    """
        Opens a csv file from the Epileptic Seizure Recognition Dataset.
        The dataset is from : https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
        and originally from : https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

        Arguments:
        ----------
        csv_file: str
            Path to the csv file of the Epileptic Seizure Recognition dataset.

        Returns:
        --------
        data_without_patients: dict
            Dictionary containing all the data in the csv file WITHOUT taking
            into account the patients. Each key correspond to the identifier
            of the sample. The value is a dictionary with two keys: "Data (EEG signal)"
            and 'Label'.
        data_per_patient: dict
            Dictionary containing the data in the csv file. Each key correspond
            to the identifier of a patient. The value is list of dictionaries
            corresponding to the EEG samples of that patient. Each sample is a
            dictionary with two keys: "Data" (EEG signal) and 'Label'. There
            are 5 labels:
                -5: eyes open, means when they were recording the EEG signal of
                    the brain the patient had their eyes open
                -4: eyes closed, means when they were recording the EEG signal
                    the patient had their eyes closed
                -3: Yes they identify where the region of the tumor was in the
                    brain and recording the EEG activity from the healthy brain area
                -2: They recorder the EEG from the area where the tumor was located
                -1: Recording of seizure activity
            All subjects falling in classes 2, 3, 4, and 5 are subjects who did not
            have epileptic seizure. Only subjects in class 1 have epileptic seizure.
        samples_per_class: dict
            Dictionary where the key is the class ID and the values is the number
            of samples for that class.
    """
    data_without_patients_list = []
    data_per_patient = {}
    patient_ids_old_new = {} # Dict of the form {old_patient_id (str): new_patient_id (int)}
    new_patient_ID = 0
    samples_per_class = {1: 0, 2: 0, 3: 0, 4:0, 5:0}
    with open(csv_file, newline='') as csvfile:
        # CSV reader
        csv_reader = csv.reader(csvfile, delimiter=',')
        # CSV important columns
        sample_name_column = 0
        label_column = -1
        # Row nb
        row_nb = 0
        for row in csv_reader:
            if (row_nb > 0):
                # ID of the patient of the sample
                if ('V' not in row[sample_name_column].split('.')[-1]):
                    patient_ID = row[sample_name_column].split('.')[-1]
                    if (patient_ID not in patient_ids_old_new):
                        patient_ids_old_new[patient_ID] = new_patient_ID
                        new_patient_ID += 1
                else:
                    patient_ID = row[sample_name_column].split('.')[-1]
                    if (patient_ID not in patient_ids_old_new):
                        patient_ids_old_new[patient_ID] = new_patient_ID
                        new_patient_ID += 1

                # Label of the sample
                label = int(row[label_column])
                #print("\tLabel: ", label)
                samples_per_class[label] += 1

                # EEG data
                eeg_data = []
                for str_val in row[1:-1]:
                    eeg_data.append(float(str_val))
                eeg_data = np.array(eeg_data)

                # Normalizing (linear scaling to range)
                eeg_data = (eeg_data - eeg_data.min())/(eeg_data.max()-eeg_data.min())
                ## Normalizing (Z-Score)
                #eeg_data = (eeg_data - eeg_data.mean())/eeg_data.std()

                # Final sample
                sample = {
                            'Data': eeg_data,
                            'Label': label
                         }

                # Adding the final sample to the dictionary of samples
                data_without_patients_list.append(sample)

                # Adding the final sample to the dictionary of samples per patient
                #if (patient_ID not in data_per_patient):
                if (patient_ids_old_new[patient_ID] not in data_per_patient):
                    #data_per_patient[patient_ID] = [sample]
                    data_per_patient[patient_ids_old_new[patient_ID]] = [sample]
                else:
                    #data_per_patient[patient_ID].append(sample)
                    data_per_patient[patient_ids_old_new[patient_ID]].append(sample)
            row_nb += 1

    # Creating a dict for all the samples without the patients
    shuffle(data_without_patients_list)
    data_without_patients = {i:data_without_patients_list[i] for i in range(len(data_without_patients_list))}

    print("Samples per class: ", samples_per_class)
    return data_without_patients, data_per_patient, samples_per_class


# Data split function without taking into account the patients
def eeg_epileptic_recog_split(data_without_patients, samples_per_class, train_prop=0.8):
    """
        Split the patients in two dictionaries of patients, one for
        training and one for testing.

        Arguments:
        ----------
        data_without_patients: dict
            Dictionary containing all the data in the csv file WITHOUT taking
            into account the patients. Each key correspond to the identifier
            of the sample. The value is a dictionary with two keys: "Data (EEG signal)"
            and 'Label'.
        samples_per_class: dict
            Dictionary where the key is the class ID and the values is the number
            of samples for that class.
        train_prop: float
            Between 0 and 1, it corresponds to the proportion of samples
            to use for the training set. By default, the proportion of
            samples to use in the testing set is (1 - train_prop).

        Returns:
        --------
        train_data: dict
            Dictionary with the same structure as data_without_patients but containing only
            the training data.
        test_data: dict
            Dictionary with the same structure as data_without_patients but containing only
            the testing data.
    """
    # Computing the test proportion of samples
    test_prop = 1 - train_prop
    nb_train_samples_per_class = {class_id:round(samples_per_class[class_id]*train_prop) for class_id in samples_per_class}
    nb_test_samples_per_class = {class_id:round(samples_per_class[class_id]*test_prop) for class_id in samples_per_class}
    print("Total number of samples per class: ", samples_per_class)
    print("Train number of samples per class: ", nb_train_samples_per_class)
    print("Test number of samples per class: ", nb_test_samples_per_class)

    # Getting the training and testing samples
    train_data_list, test_data_list = [], []
    current_nb_train_samples_per_class = {1: 0, 2: 0, 3: 0, 4:0, 5:0}
    current_nb_test_samples_per_class = {1: 0, 2: 0, 3: 0, 4:0, 5:0}
    for sample_ID in data_without_patients:
        # Getting the sample
        sample = data_without_patients[sample_ID]

        # Determining its split
        if (current_nb_train_samples_per_class[sample['Label']] < nb_train_samples_per_class[sample['Label']]):
            train_data_list.append(sample)
            current_nb_train_samples_per_class[sample['Label']] += 1
        else:
            test_data_list.append(sample)
            current_nb_test_samples_per_class[sample['Label']] += 1

    # Shuffling the splits
    shuffle(train_data_list)
    shuffle(test_data_list)

    # Getting the training and testing data dicts
    train_data = {i:train_data_list[i] for i in range(len(train_data_list))}
    test_data = {i:test_data_list[i] for i in range(len(test_data_list))}

    print("\nNumber of training samples: ", len(train_data))
    print("Number of training samples per class: ", current_nb_train_samples_per_class)
    print("\nNumber of testing samples: ", len(test_data))
    print("Number of testing samples per class: ", current_nb_test_samples_per_class)
    return train_data, test_data


# HDF5 file creation (necessary for reproducibility)
def hdf5FileCreation_EEG(
                        dataset_folder='../../data/EEG_Epileptic_Seizure_Recognition/'
                    ):
    """
        Creates a hdf5 file called "data.hdf5" containing the split of the data
        between train and test. The structure of the HDF5 file is the following:
            Split (train or test)
                Label
                Data

        Parameters:
        -----------
        dataset_folder: str
            Path to the folder containing the data
    """
    # Putting the dataset_folder under the right format
    if (dataset_folder[-1] != '/'):
        dataset_folder += '/'

    # Training and testing samples
    # train_prop = 0.8
    train_prop = 0.9
    csv_dataset_file = dataset_folder + "/Epileptic Seizure Recognition.csv"
    data_without_patients, data_per_patient, samples_per_class = open_csv_eeg_epileptic_recognition_dataset(csv_dataset_file)
    train_samples, test_samples = eeg_epileptic_recog_split(data_without_patients, samples_per_class, train_prop=train_prop)

    # Some statistics about the dataset
    print("=======> Number of testing samples per class")
    print("\t", samples_per_class)

    print("{} % of the samples are going to be used for training and {} % of the samples for testing".format( len(train_samples)/(len(train_samples)+len(test_samples)), len(test_samples)/(len(train_samples)+len(test_samples)) ))

    # Creating an hdf5 file for the dataset
    hdf5_file = h5py.File(dataset_folder+"/data.hdf5", "w")
    dataset = hdf5_file.create_group("mydataset")
    train_dataset = hdf5_file.create_group('mydataset/train')
    test_dataset = hdf5_file.create_group('mydataset/test')

    # Training dataset
    train_id = 0
    for train_sample_id in train_samples:
        train_sample = train_samples[train_sample_id]
        sample = train_dataset.create_group(str(train_id))

        sample.attrs['Label'] = train_sample["Label"]
        sample.attrs['Data'] = train_sample["Data"]

        train_id += 1

    # Testing dataset
    test_id = 0
    for test_sample_id in test_samples:
        test_sample = test_samples[test_sample_id]
        sample = test_dataset.create_group(str(test_id))

        sample.attrs['Label'] = test_sample["Label"]
        sample.attrs['Data'] = test_sample["Data"]

        test_id += 1


# HDF5 file loading (necessary for reproducibility)
def loadFromHDF5_EEG(hdf5_file_path='../../data/EEG_Epileptic_Seizure_Recognition/data.hdf5'):
    """
        Create two dictionaries containing the training and test datasets

        Parameters:
        -----------
            hdf5_file_path: str
                Path to an hdf5 file containig the structure of the data to use

        Returns:
        --------
        train_data: dict
            Dictionary where the key is the id of a sample and the value is
            another dictionary with two keys 'Label' and 'Data'
        test_data: dict
            Dictionary where the key is the id of a sample and the value is
            another dictionary with two keys 'Label' and 'Data'
    """
    # Creating the dictionaries for the dataset
    train_data, test_data = {}, {}

    # Loading the data
    h5f = h5py.File(hdf5_file_path, 'r')
    for dataset in h5f:
        # print("Dataset: ", dataset)
        for data_split in h5f[dataset]:
            # print("Data split: ", data_split)
            for sampleID in h5f[dataset][data_split]:
                # print("sampleID: ", sampleID)
                if (data_split == 'train'):
                    if (int(sampleID) in train_data):
                        print("In the training split, the sample of ID {} has already been stored".format(sampleID))
                    else:
                        train_data[int(sampleID)] = {
                                                    'Label': h5f[dataset][data_split][sampleID].attrs['Label'],\
                                                    'Data': h5f[dataset][data_split][sampleID].attrs['Data']
                                                }
                elif (data_split == 'test'):
                    if (int(sampleID) in test_data):
                        print("In the testing split, the sample of ID {} has already been stored".format(sampleID))
                    else:
                        test_data[int(sampleID)] = {
                                                    'Label': h5f[dataset][data_split][sampleID].attrs['Label'],\
                                                    'Data': h5f[dataset][data_split][sampleID].attrs['Data']
                                                }
                else:
                    raise ValueError("Data split {} not recognized".format(data_split))

    # Closing file
    h5f.close()

    # Verifying the number of samples per split
    print("Number of samples for training: ", len(train_data))
    print("Number of samples for testing: ", len(test_data))

    return train_data, test_data


# Creating the new dataste class
class EEG_EpilepticSeizureRecognition(Dataset):
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
        self.sample_rate = 4907/23.5 # Hz
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

        # If binary classification, we do 1 vs {2, 3, 4, 5} which corresponds
        # to Epileptic Seizeure against the rest (commonly done when people
        # use this dataset)
        if (self.params['binarizeDS']):
            # WARNING: IN THIS CASE THE LABEL OF Epileptic Seizure is 0 and not 1
            if (label == 1):
                label = 0
            else:
                label = 1
        else:
            # WARNING: IN THIS CASE THE LABEL OF Epileptic Seizure is 0 and not 1
            label -= 1

        return torch.from_numpy(feature.copy()).float(), label


# Creating the new dataste class
class EEG_EpilepticSeizureRecognition_Multifeature(Dataset):
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
        self.sample_rate = 4907/23.5 # Hz
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

        # If binary classification, we do 1 vs {2, 3, 4, 5} which corresponds
        # to Epileptic Seizeure against the rest (commonly done when people
        # use this dataset)
        if (self.params['binarizeDS']):
            # WARNING: IN THIS CASE THE LABEL OF Epileptic Seizure is 0 and not 1
            if (label == 1):
                label = 0
            else:
                label = 1
        else:
            # WARNING: IN THIS CASE THE LABEL OF Epileptic Seizure is 0 and not 1
            label -= 1

        return features, label


###############################################################################
###############################################################################

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--dataset_folder", required=True, default='../../data/EEG_Epileptic_Seizure_Recognition/', help="Folder containing the data", type=str)
    args = vars(ap.parse_args())

    # Getting the arguments
    dataset_folder = args['dataset_folder']

    #==========================================================================#
    # Creating the HDF5 files
    hdf5FileCreation_EEG(dataset_folder=dataset_folder)

    #==========================================================================#


if __name__=="__main__":
    main()
