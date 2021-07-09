#!/usr/bin/env python
# coding: utf-8

# # Imports


import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion # for local neighborhoof
from scipy.ndimage.morphology import iterate_structure # for local neighborhood

from numba import njit
from typing import Tuple, Callable, List
from typing import List, Tuple, Dict
from microphone import record_audio

import pickle


fingerprint_database = dict()
song_database = []


# # Custom Classes

class FingerprintKey:
    def __init__(self, key, ti):
        self.key = key # (fi, fj, delta_t)
        self.ti = ti # Absolute start point

class SongRecord:
    def __init__(self, song, ti):
        self.song = song
        self.ti = ti # Relative (Original_song.ti - Sample_Recording.ti)
    
    def __repr__(self):
        return "(" + str(self.song) + " " + str(self.ti) + ")"


# # Peak Finding Functions


# `@njit` "decorates" the `_peaks` function. This tells Numba to
# compile this function using the "low level virtual machine" (LLVM)
# compiler. The resulting object is a Python function that, when called,
# executes optimized machine code instead of the Python code
# 
# The code used in _peaks adheres strictly to the subset of Python and
# NumPy that is supported by Numba's jit. This is a requirement in order
# for Numba to know how to compile this function to more efficient
# instructions for the machine to execute
@njit
def _peaks(
    data_2d: np.ndarray, rows: np.ndarray, cols: np.ndarray, amp_min: float
) -> List[Tuple[int, int]]:
    """
    A Numba-optimized 2-D peak-finding algorithm.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.

    rows : numpy.ndarray, shape-(N,)
        The 0-centered row indices of the local neighborhood mask
    
    cols : numpy.ndarray, shape-(N,)
        The 0-centered column indices of the local neighborhood mask
        
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location. 
    """
    peaks = []  # stores the (row, col) locations of all the local peaks

    # Iterate over the 2-D data in col-major order
    # we want to see if there is a local peak located at
    # row=r, col=c

    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            # The amplitude falls beneath the minimum threshold
            # thus this can't be a peak.
            continue
        
        # Iterating over the neighborhood centered on (r, c)
        # dr: displacement from r
        # dc: discplacement from c
        for dr, dc in zip(rows, cols):
            if dr == 0 and dc == 0:
                # This would compare (r, c) with itself.. skip!
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                # neighbor falls outside of boundary
                continue

            # mirror over array boundary
            if not (0 <= c + dc < data_2d.shape[1]):
                # neighbor falls outside of boundary
                continue

            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                # One of the amplitudes within the neighborhood
                # is larger, thus data_2d[r, c] cannot be a peak
                break
        else:
            # if we did not break from the for-loop then (r, c) is a peak
            peaks.append((r, c))
    return peaks

# `local_peak_locations` is responsible for taking in the boolean mask `neighborhood`
# and converting it to a form that can be used by `_peaks`. This "outer" code is 
# not compatible with Numba which is why we end up using two functions:
# `local_peak_locations` does some initial pre-processing that is not compatible with
# Numba, and then it calls `_peaks` which contains all of the jit-compatible code
def local_peak_locations(data_2d: np.ndarray, neighborhood: np.ndarray, amp_min: float):
    """
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the specified `amp_min`.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected
    
    neighborhood : numpy.ndarray, shape-(h, w)
        A boolean mask indicating the "neighborhood" in which each
        datum will be assessed to determine whether or not it is
        a local peak. h and w must be odd-valued numbers
        
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location.
    
    Notes
    -----
    Neighborhoods that overlap with the boundary are mirrored across the boundary.
    
    The local peaks are returned in column-major order.
    """
    
    rows, cols = np.where(neighborhood)
    
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1

    # center neighborhood indices around center of neighborhood
    rows -= neighborhood.shape[0] // 2
    
    cols -= neighborhood.shape[1] // 2
    

    return _peaks(data_2d, rows, cols, amp_min=amp_min)

def local_peaks_mask(data: np.ndarray, cutoff: float) -> np.ndarray:
    """Find local peaks in a 2D array of data.

    Parameters
    ----------
    data : numpy.ndarray, shape-(H, W)

    cutoff : float
         A threshold value that distinguishes background from foreground

    Returns
    -------
    Binary indicator, of the same shape as `data`. The value of
    1 indicates a local peak."""
    # Generate a rank-2, connectivity-2 binary mask
    neighborhood_mask = iterate_structure(generate_binary_structure(rank = 2, connectivity = 1), 20)  # <COGLINE>
    

    # Use that neighborhood to find the local peaks in `data`.
    # Pass `cutoff` as `amp_min` to `local_peak_locations`.
    peak_locations = local_peak_locations(data, neighborhood_mask, cutoff)  # <COGLINE>
   
    

    # Turns the list of (row, col) peak locations into a shape-(N_peak, 2) array
    # Save the result to the variable `peak_locations`
    #peak_locations = np.array(peak_locations)

    # create a mask of zeros with the same shape as `data`
    
    #mask = np.zeros(data.shape, dtype=bool)

    # populate the local peaks with `1`
    #mask[peak_locations[:, 0], peak_locations[:, 1]] = 1
    return peak_locations


# # Database Creation Functions


def load_audio(song_name):
    '''
    Return the recorded audio and sampling_rate as a Tuple
    '''
    #length = 60
    length = 10
    return librosa.load(song_name, sr = 44100, duration = length)




def get_spectrogram(recorded_audio, sampling_rate):
    '''
    Return the spectrogram corresponding to `recorded_audio` with `sampling_rate` as the sampling_rate
    '''
   
    return mlab.specgram(recorded_audio,
                            NFFT=4096,
                            Fs=sampling_rate,
                            window=mlab.window_hanning,
                            noverlap=4096//2,
                            mode='magnitude')

# make list "peaks" of all peaks, iterate through them with a fanout value and constructa fingerprint
def construct_fingerprint(peaks: List[Tuple[int, int]]) -> List[FingerprintKey]:
    
    #IMPORTANT: need to find the index for the first peak in our temporal window and start iterating from there

    """
    Parameters
    ----------
    peak: List of tuples for peak location, the first value 
    being the peak and the second value being 
    the time at which it occured. 

    peak is the output of  local_peak_locations

    Returns
    -------
    A list of FingerPrintKey objects that make up the whole fingerprint for some peak. This list will be stored in
    the database through the function put_database

    """

    fanout = 15
    fingerprint = []
    #start_index = 0 #need to find the index for the first peak in our temporal window and start iterating from there
    
    # find location of first peak in the list
    
    for start_index in range(0, len(peaks)-fanout-1): 
        for i in range(start_index + 1, start_index + fanout + 1):

            # compare peak time to the next 15 peaks
            abs_time = peaks[start_index][1] # the time of the first peak recorded
            peak, time = peaks[i]
            next_peak, next_time = peaks[i + 1]
            _print = FingerprintKey((peak, next_peak, next_time - abs_time), abs_time)
            fingerprint.append(_print)
    
    return fingerprint 

def put_database(fingerprint_database: Dict[FingerprintKey, List[SongRecord]], song_database: List[SongRecord], fingerprint: FingerprintKey, song: SongRecord):
    '''
    Put a song and fingerprint in the database
    '''
    if fingerprint.key not in fingerprint_database:
        fingerprint_database[fingerprint.key] = []
    fingerprint_database[fingerprint.key].append(song)
    song_database.append(song)

def fill_database(song_names: List[str], fingerprint_database: Dict[FingerprintKey, SongRecord], song_database: List[SongRecord]):
    '''
    Fill the database given the fingerprint database, song database,
    and a list of song names (file path to songs).
    '''
    for s in range(len(song_names)):
        recorded_audio, sampling_rate = load_audio(song_names[s])
        spectrogram, freqs, times = get_spectrogram(recorded_audio, sampling_rate)
        cutoff = get_cutoff(spectrogram)
        peaks = local_peaks_mask(spectrogram, cutoff)
        fingerprint_keys = construct_fingerprint(peaks)
        for fingerprint in fingerprint_keys:
            sr = SongRecord(s, fingerprint.ti)
            put_database(fingerprint_database, song_database, fingerprint, sr)

def get_cutoff(S):
    '''
    Get cutoff of spectrogram
    '''
    
    S = S.ravel() # flatten array
    ind = round(len(S) * 0.75)
    cutoff_log_amplitude = np.partition(S, ind)[ind]
    return cutoff_log_amplitude
    


# # Database Creation Process

# ### Song Names Defining

# In[15]:


song_names = ["Blah, Blah, Blah", "Hotel California", "Life In The Fast Lane", "Billie Jean", "Good 4 U",
              "Rasputin", "Walking On The Sun", "You Belong With Me", "Our Song", "Turn Back Time", "Good Life",
              "Stitches", "Breakeven"]

name_to_artist = {"Blah, Blah, Blah" : "Armin van Buuren", "Hotel California" : "Eagles", "Life In The Fast Lane" : "Eagles",
                  "Billie Jean" : "Michael Jackson", "Good 4 U" : "Olivia Rodrigo", "Rasputin" : "Boney M.",
                  "Walking On The Sun" : "Smashmouth", "You Belong With Me" : "Taylor Swift", "Our Song" : "Anne_marie, Niall Horan",
                  "Turn Back Time" : "Daniel Schulz", "Good Life" : "OneRepublic", "Stitches" : "Shawn Mendes", "Breakeven" : "The Script"}

name_to_file = {"Blah, Blah, Blah" : "blah_blah_blah.m4a", "Hotel California" : "hotel_california.mp3",
                "Life In The Fast Lane" : "life_in_the_fast_lane.mp3",
                  "Billie Jean" : "billie_jean.mp3", "Good 4 U" : "good_4_u.mp3",
                "Rasputin" : "rasputin.mp3",
                  "Walking On The Sun" : "walking_on_the_sun.mp3", 
                "You Belong With Me" : "you_belong_with_me.mp3",
                "Our Song" : "our_song.mp3",
                  "Turn Back Time" : "turn_back_time.mp3",
                "Good Life" : "good_life.mp3",
                "Stitches" : "stitches.mp3",
                "Breakeven" : "breakeven.mp3"}

# ### Saving Database into a File


# assuming 'database' has already been created
def create_database():
    with open("database.pkl", "wb") as file:
        pickle.dump(fingerprint_database, file)


# ### Loading Database for Use

# In[18]:


def load_database():
    with open("database.pkl", "rb") as file:
        fingerprint_database = pickle.load(file)
        return fingerprint_database


# # Obtaining User Recording


def listen_audio():
    listen_time = 10
    frames, sample_rate = record_audio(listen_time)
    recorded_audio = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    return (recorded_audio, sample_rate)


# ### Compute user recording --> fingerprints


def compute_fingerprint(recorded_audio, sampling_rate):
    spectrogram, freq, times = get_spectrogram(recorded_audio, sampling_rate)
    cutoff = get_cutoff(spectrogram)
    peaks = local_peaks_mask(spectrogram, cutoff)
    fingerprint_keys = construct_fingerprint(peaks)
    return fingerprint_keys


# # Compare Sample_Fingerprints with Fingerprint Database

# In[16]:


from collections import Counter

def compare(sample_fingerprints, database, threshold = 0):
    '''
    Takes sample_fingerprints and matches them with database.
    Returns song that it found or None if no song was found.
    
    sample_fingerprints: A list of FingerprintKey objects for the song sample
    
    database: The database (dictionary) of fingerprints to compare, with the keys are FingerprintKey objects and value of SongRecord objects
    '''
    #make song_counter
    song_counter = Counter()
    for fingerprint in sample_fingerprints:
        #print("Fingerprint:", fingerprint)
        #print("Found Finger:", )
        if fingerprint.key in database: #if a match is found
            possible_songs = database[fingerprint.key]
            song_records = []
            for song_record in possible_songs: # put all the possible songs in the database
                new_song_record = (song_record.song, song_record.ti - fingerprint.ti)
                song_records.append(new_song_record)
            song_counter.update(song_records)
    #print("Song Tally:", song_counter)
    try:
        result, tally = song_counter.most_common(1)[0]
        #print("Highest Tally:", tally)
        #print("Corresponding Song_Record object:", result)
        if tally >= threshold:
            return result[0] # return song_id
        else:
            return "Song Not Found!"
    except (Exception, IndexError):
        print("Song Not Found!")

