"""
We find it hard to map Dexed's 0-99 output level to actual amplitude.
So we conducted an empirical experiment, and manually fit the values using np.polyfit
RMS is xx. Details to be released.
"""
import numpy as np

def dexed_ol_to_amplitude(x):
    return 4e-4 * np.exp(0.086 * x)

def amplitude_to_dexed_ol(x):
    return int((np.log(x) - np.log(4e-4)) / 0.086)