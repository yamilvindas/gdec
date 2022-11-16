#!/usr/bin/env python3

import os
import csv
import h5py

import librosa

import wave
import torch
import torchaudio

import numpy as np

from src.utils.spectrogram import stft_signal, estimate_inf_sup_db

def get_information_wav(wav_path):
    """
        Returns some useful information about a wav file

        Arguments:
        ----------
        wav_path: str
            Path to the wav file that we want to analyze

        Returns:
        --------
        channels: int
            Number of channels in the input audio
        SampleRate: float
        bit_type: int
            Number of bits used to encoded the amplitude values of the audio
        frames: int
            Number of samples in the audio
        duration: float
    """
    # Loading the file
    f = wave.open(wav_path)

    #Audio head parameters
    params = f.getparams()
    channels = f.getnchannels()
    sampleRate = f.getframerate()
    bit_type = f.getsampwidth() * 8
    frames = f.getnframes()
    duration = frames / float (sampleRate)
    f.close()

    return channels, sampleRate, bit_type, frames, duration

def get_feature(
                        signal,
                        sample_rate,
                        add_channel_dim,
                        feature_type,
                        params
                    ):
    """
        Loads an audio feature from a sample.

        Arguments:
        -----------
        sample: list, np.array
            List of values constituting the ECG signal
        sample_rate: int
            Sample rate of the raw signal
        feature_type: str
            Time-frequency representation to use. Different options are possible:
                - RawSignal
                - RawSignalSegmented
                - Spectrogram
                - MelSpectrogram
                - MFCC
                - Cochleagram
        add_channel_dim: bool
            True if we want to add a dimension for the channel. This is useful
            when using 2D convolutions.
        params: dict
            Parameters to compute the selected representation. This parameters
            vary from one representation to another

        Returns:
        --------
        feature: np.array
            Audio feature
    """
    # Getting the desired feature
    if (feature_type.lower() == 'rawsignal'):
        signal = signal.reshape((1, signal.shape[0]))
        feature = signal
    elif (feature_type.lower() == 'spectrogram'):
        # Parameters for the log spectrogram
        n_fft = params['n_fft']
        hop_length = params['n_overlap']
        window = params['window']
        # Computation of the log spectrogram
        stft_waveform, f, t = stft_signal(signal, n_fft, sample_rate, hop_length, window)
        spectrogram = np.absolute(stft_waveform)**2
        log_spec = 20*np.log10(spectrogram)
        # Estimating the min and max db values of the spectrogram in log scale for the colobar
        x, y = 0, 9
        min_db, max_db = estimate_inf_sup_db(spectrogram, x, y)
        # Filtering the spectrogram
        log_spec = 20*np.log10(spectrogram)
        log_spec[log_spec < min_db] = min_db
        log_spec[log_spec > max_db] = max_db
        feature = log_spec
    else:
        raise NotImplementedError("Feature {} is not implemented".format(feature_type))

    # Adding channel dim if necessary
    if (add_channel_dim):
        if (feature_type.lower() != 'rawsignal'):
            feature = np.reshape(feature, (1, feature.shape[0], feature.shape[1])) # We have
            # to do this in order to be able to apply 2D convolutions

    return feature
