#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt

import librosa
import wave

import torch
from nnAudio import features

import math
import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift

def stft_signal(input_signal, nfft, sample_rate, noverlap, window):
    """
        Compute the Short Time Fourier Transform (code based
        on Kevin Guépies matlab code)

        Arguments:
        ----------
        input_signal: np.array
            Input signal from which we want to compute the STFT
        nfft: int
            Number of points to use to compute the STFT
        sample_rate: float
            Sample frequency of the input signal.
        noverlap: int
            Number of points to overlap between segments. Not necessarily equal to hop_length???
        window: str
            Window to use to compute the spectrogram

        Returns:
        --------
        stft_input_signal: np.array
            Short Time Fourier Transform of the input signal.
        f: np.array or list
            Frequency vector in Hz.
        t: np.array or list
            Time vector in s
    """
    # Treating the multi-channel input and computing its length
    if (len(input_signal.shape) > 1):
        if (input_signal.shape[0] > 1):
            np.transpose(input_signal)
            signal_lenght = input_signal.shape[1]
    else:
        signal_lenght = input_signal.shape[0]

    # Create the window
    win = tftb_window(nfft, window)

    # Creating the STFT matrix
    h = noverlap # Hop size
    nb_rows = nfft
    nb_cols = 1+(signal_lenght-nfft)//h
    stft_input_signal = np.zeros((nb_rows, nb_cols))

    # Filling the values of the STFT
    idx, col = 0, 0
    while (idx + nfft < signal_lenght):
        # Windowing
        windowed_signal = np.multiply(input_signal[idx:idx+nfft], win) # Element-wise multiplication

        # FFT
        fft_input_signal = fft(windowed_signal, nfft)
        fft_input_signal = fftshift(fft_input_signal) # Shift the zero-frequency component to
        # the center of the spectrum. This allow to have positive and negative frequencies

        # Uptade the STFT matrix
        stft_input_signal[:, col] = fft_input_signal

        # Update the indexes
        idx += h
        col += 1

    # Frequencies
    f = np.arange(start=math.floor(-nb_rows/2), stop=math.floor(nb_rows/2))*(sample_rate/nfft)
    #f = np.arange(start=math.floor(0), stop=math.floor(nb_rows))*(sample_rate/nfft)

    # Times
    t = np.arange(start=0, stop=nb_cols*h, step=h)/sample_rate


    return stft_input_signal, f, t

def tftb_window(length, window_type):
    """
        Creates a window of lenght nfft

        Arguments:
        ----------
        lenght: int
            Lenght of the window
        window_type: str
            Window type

        Returns:
        --------
        window: np.array
    """
    if (window_type.lower() == 'hamming'):
        window = signal.windows.hamming(length)
    elif (window_type.lower() == 'hann'):
        window = signal.windows.hann(length)
    elif (window_type.lower() == 'blackman'):
        window = signal.windows.blackman(length)
    elif (window_type.lower() == 'triang'):
        window = signal.windows.triang(length)
    elif (window_type.lower() == 'boxcar'):
        window = signal.windows.boxcar(length)
    elif (window_type.lower() == 'bartlett'):
        window = signal.windows.bartlett(length)
    elif (window_type.lower() == 'flattop'):
        window = signal.windows.flattop(length)
    elif (window_type.lower() == 'parzen'):
        window = signal.windows.parzen(length)
    elif (window_type.lower() == 'bohman'):
        window = signal.windows.bohman(length)
    elif (window_type.lower() == 'blackmanharris'):
        window = signal.windows.blackmanharris(length)
    elif (window_type.lower() == 'nuttall'):
        window = signal.windows.nuttall(length)
    elif (window_type.lower() == 'barthann'):
        window = signal.windows.barthann(length)
    elif (window_type.lower() == 'cosine'):
        window = signal.windows.cosine(length)
    elif (window_type.lower() == 'exponential'):
        window = signal.windows.exponential(length)
    elif (window_type.lower() == 'tukey'):
        window = signal.windows.tukey(length)
    elif (window_type.lower() == 'taylor'):
        window = signal.windows.taylor(length)
    else:
        raise ValueError("Windows type {} not supported".format(window_type))
    return window

def estimate_inf_sup_db(spectrogram, x, y):
    """
        Estimates the min and max db values of the spectrogram.
        This is useful to define the min and max values of the colorbar
        for plotting purposes.

        Arguments:
        ----------
        spectrogram: np.array
            Spectrogram used from which we want to compute its min and
            max values
        x: int
            Value to detect the inf in order to threshold the time-frequency representation by the lower values
        y: int
            Value to detect the sup. Threshold of the time-frequency representation is not necessary. The
            sup is only useful for the colorbar for plotting purposes

        Returns:
        --------
        min_db: float
        max_db: float
    """
    spectrogram_db = 20*np.log10(spectrogram)
    # Because we compute a log, we have to remove the -inf values by setting them to 0 (for instance)
    for i in range(spectrogram_db.shape[0]):
        for j in range(spectrogram_db.shape[1]):
            if (spectrogram_db[i][j] == -math.inf):
                spectrogram_db[i][j] = 0.
    moy_spectrogram = np.mean(spectrogram_db)
    std_spectrogram = np.std(spectrogram_db)
    min_db = moy_spectrogram + x*std_spectrogram
    max_db = moy_spectrogram + y*std_spectrogram
    return min_db, max_db
