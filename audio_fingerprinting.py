import librosa
import numpy as np 
from skimage.feature import peak_local_max 

def get_coordinates_from_audio_file(audio_path: str, config: dict, debug=False):
    """
    Load audio and extract constellation map peak coordinates.
    
    Args:
        audio_path (str): Path to the audio file to load.
        debug (bool): If True, print debug information about peaks and spectrogram shape.
    
    Returns:
        tuple: (coordinates, stft) where:
            - coordinates (ndarray): Peak locations sorted by time, shape (num_peaks, 2) with columns [frequency_bin, time_frame]
            - stft (ndarray): The spectrogram representation, shape (num_freq_bins, num_time_frames)
    """
    y, sr = librosa.load(audio_path, sr=22050)
    if debug: 
        # Report the file size and sampling context when debugging.
        print(f"Loaded audio file: {audio_path}")
        print(f"Audio duration: {len(y)/sr:.2f} seconds, Sample rate: {sr} Hz")
    
    n_mels = 512    
    n_fft = 1024
    win_length = n_fft
    hop_length = int(win_length / 2)
    cqt_n_bins = 128
    cqt_bins_per_octave = 24
    
    # Linear STFT
    stft = np.abs(librosa.stft(y, n_fft=n_fft, window='hann', win_length=win_length, hop_length=hop_length))
    # MEL
    # stft = np.abs(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length))
    
    # MFCC
    # stft = np.abs(librosa.feature.mfcc(y=y, n_mfcc=n_fft, sr=sr))
    
    # CQT
    # stft = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=cqt_n_bins, bins_per_octave=cqt_bins_per_octave))
    # coordinates = peak_local_max(np.log1p(stft), min_distance=10,threshold_rel=0.05)
    
    coordinates = peak_local_max(
        np.log1p(stft + 1e-8), 
        min_distance=config["min_distance"],
        threshold_rel=config["threshold_rel"],
        threshold_abs=config["threshold_abs"],
        exclude_border=config["exclude_border"],
        num_peaks=config["num_peaks"] * (stft.shape[0] * stft.shape[1]),  # Adjust based on spectrogram size
    )
    coordinates = coordinates[coordinates[:,1].argsort()]
    
    return coordinates, stft

def generate_hashes(
    coordinates: np.ndarray,
    stft: np.ndarray,
    fan_out: int = 15,
    min_dt: int = 1,
    max_dt: int = 200,
    min_df: int = 0,
    max_df: int = 100,
    debug: bool = False,
):
    """
    Generate anchor-target hash pairs from constellation map peaks.
    
    Creates hash tuples from chronologically adjacent peaks using fan-out pairing.
    Each anchor peak is paired with up to fan_out subsequent peaks in time.
    
    Args:
        coordinates (ndarray): Peak locations, shape (num_peaks, 2) with columns [frequency_bin, time_frame].
        stft (ndarray): Spectrogram for shape information, shape (num_freq_bins, num_time_frames).
        fan_out (int): Number of target pairs per anchor (default 15).
        min_dt (int): Minimum time difference (in frames) between anchor and target.
        max_dt (int): Maximum time difference (in frames) between anchor and target.
        min_df (int): Minimum absolute frequency-bin difference between anchor and target.
        max_df (int): Maximum absolute frequency-bin difference between anchor and target.
        debug (bool): If True, print debug information about peaks and pairing.
    
    Returns:
        list: Hash tuples of the form ((anchor_freq, target_freq, time_delta), anchor_time).
              Each hash key is a 3-tuple: (anchor frequency bin, target frequency bin, time difference).
    """
    hashes = []
    num_peaks = len(coordinates)
    
    if debug:
        print(f"Peak coordinates for: {len(coordinates)}")
        print(f"Number of frequency bins: {stft.shape[0]}, Number of time frames: {stft.shape[1]}")

    for i in range(num_peaks):
        anchor_k, anchor_n = coordinates[i]
        used = 0
        
        for f in range(i + 1, num_peaks):
            tgt_k, tgt_n = coordinates[f]
            n_diff = tgt_n - anchor_n

            if n_diff < min_dt:
                continue
            if n_diff > max_dt:
                break

            k_diff = abs(tgt_k - anchor_k)
            if k_diff < min_df or k_diff > max_df:
                continue

            hashes.append(((int(anchor_k), int(tgt_k), int(n_diff)), int(anchor_n)))
            used += 1
            if used >= fan_out:
                break
    return hashes

def get_file_matches(query_hashes: dict, fingerprints: dict):
    """
    Count hash matches between query and database fingerprints.
    
    Counts how many query hashes match hashes in the fingerprint database,
    aggregating matches by database file.
    
    Args:
        query_hashes (list): Hash tuples from query audio, format: ((anchor_freq, target_freq, time_delta), anchor_time).
        fingerprints (dict): Inverted index mapping hash tuples to list of (file, anchor_time) postings.
    
    Returns:
        dict: Match counts keyed by database filename, values are match counts.
    """
    match_counts = {}
    for query_hash, query_anchor_n in query_hashes:
        if query_hash in fingerprints:
            for db_file, db_anchor_n in fingerprints[query_hash]:
                if db_file not in match_counts:
                    match_counts[db_file] = 0
                match_counts[db_file] += 1
    return match_counts

