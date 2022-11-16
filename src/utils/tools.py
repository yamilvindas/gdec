#!/usr/bin/env python3

import os
import csv
import h5py

from collections import Counter

from random import shuffle

import numpy as np

import torch

from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def string_to_bool(string):
    if (type(string) == bool):
        return string
    else:
        if (string.lower() == 'true'):
            return True
        elif (string.lower() == 'false'):
            return False
        else:
            raise ValueError("String {} is not valid to be transformed into boolean".format(string))


def train_val_split_stratified(data, test_size, n_splits):
    """
        Split the data into two stratified splits, one for training and one for
        validation.

        Arguments:
        ----------
        data: dict
            Containing the different samples of the dataset.
        test_size: float
            Percentage of samples to be used as test.
        n_splits: int
            Number of splits to create. Interesting if cross-validation is used
            as evaluation metric

        Returns:
        --------
        splits: list
            List where each element correspond to a train/validation split.
            Each element is a dict with two keys, 'Train' and 'Validation'
            corresponding to the train and validation samples
    """
    # Getting the indices of the samples and the labels. This will allow us
    # to easily create the train and validation splits
    samples_ids = []
    samples_labels = []
    for sample_id in data:
        # Adding the sample ID
        samples_ids.append(sample_id)

        # Getting the label
        label = data[sample_id]['Label']
        samples_labels.append(label)

    # Creating the instance of stratified sampling
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    # Getting the different splits
    splits = []
    for train_idxs, val_idxs in sss.split(samples_ids, samples_labels):
        # Creating the variable for the current split
        split = {'Train': {}, 'Validation': {}}

        # Getting the training samples
        new_train_id = 0
        for train_idx in train_idxs:
            train_sample_id = samples_ids[train_idx]
            split['Train'][new_train_id] = data[train_sample_id]
            new_train_id += 1

        # Getting the validation samples
        new_val_id = 0
        for val_idx in val_idxs:
            val_sample_id = samples_ids[val_idx]
            split['Validation'][new_val_id] = data[val_sample_id]
            new_val_id += 1

        # Adding the split to the list of splits
        splits.append(split)

    return splits
