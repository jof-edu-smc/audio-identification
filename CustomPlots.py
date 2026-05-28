import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchaudio
from torchaudio.transforms import Resample, Spectrogram, TimeStretch, TimeMasking, FrequencyMasking, MelScale
import numpy as np
import librosa
import librosa.display
import os
from IPython.display import display, Audio
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np  
import sys

    
def save_spectrogram(specgram, coordinates, title=None, ylabel="freq_bin", xlabel="time", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if specgram.ndim == 3:
        specgram = specgram[0]

    db_spec = librosa.amplitude_to_db(np.abs(specgram), ref=np.max)
    ax.imshow(db_spec, origin="lower", aspect="auto", interpolation="nearest")
    
    plt.savefig("spectrogram_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    plt.savefig("peak_coordinates.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_spectrogram(specgram, coordinates, title=None, ylabel="freq_bin", xlabel="time", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if specgram.ndim == 3:
        specgram = specgram[0]

    db_spec = librosa.amplitude_to_db(np.abs(specgram), ref=np.max)
    ax.imshow(db_spec, origin="lower", aspect="auto", interpolation="nearest")
    
    plt.show()
    plt.figure(figsize=(10, 5))
    if coordinates is not None and len(coordinates) > 0:
        plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        plt.show()
        
        
def denoise_query(
        query_coordinates, 
        query_stft, 
        debug=False
    ):
    """
    Denoise query spectrogram using low-rank NMF reconstruction.
    
    Applies non-negative matrix factorization to suppress noise while preserving
    core structure, then extracts peaks from the denoised spectrogram.
    
    Args:
        query_coordinates (ndarray): Original query peak coordinates for reference.
        query_stft (ndarray): Query spectrogram to denoise, shape (num_freq_bins, num_time_frames).
        debug (bool): If True, print debug information and visualize spectrograms.
    
    Returns:
        tuple: (denoised_coordinates, denoised_stft) where:
            - denoised_coordinates (ndarray): Peaks from denoised spectrogram, sorted by time.
            - denoised_stft (ndarray): Reconstructed low-rank spectrogram after NMF.
    """
    # Use low-rank NMF reconstruction to suppress noise in magnitude spectrograms.
    # Choose a conservative low-rank factorization size from the spectrogram dimensions.
    n_components = max(2, min(64, min(query_stft.shape) - 1))
    # Configure the NMF model used for denoising.
    query_nmf = NMF(
        n_components=n_components,
        init=None, 
        solver='cd', 
        beta_loss='frobenius', 
        tol=0.0001, 
        max_iter=400,
        random_state=0,
        alpha_W=0.0, 
        alpha_H='same', 
        l1_ratio=0.0,
        shuffle=False
    )
    
    # Visualize the original query before denoising.
    plot_spectrogram(query_stft, query_coordinates, title="Original Query Spectrogram")
    print("Fitting NMF to query spectrogram for denoising...")
    # Factorize the magnitude spectrogram into low-rank components.
    W = query_nmf.fit_transform(query_stft)
    # Retrieve the learned basis vectors.
    H = query_nmf.components_
    # Reconstruct the smoothed spectrogram.
    denoised_stft = W @ H

    # Extract peaks from the denoised reconstruction.
    denoised_coordinates = peak_local_max(np.log1p(denoised_stft), min_distance=10,threshold_rel=0.05)
    # Sort peaks chronologically for consistent downstream pairing.
    denoised_coordinates = denoised_coordinates[denoised_coordinates[:, 1].argsort()]
    # Visualize the denoised result.
    plot_spectrogram(denoised_stft, denoised_coordinates, title="Denoised Query Spectrogram")

    if debug:
        # Report how many peaks survived denoising.
        print(f"Denoised coordinates: {len(denoised_coordinates)}")
    return denoised_coordinates, denoised_stft
