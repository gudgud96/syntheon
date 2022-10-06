"""
For loading and preprocessing audio
"""
import numpy as np
import torch
from syntheon.inferencer.vital.models.core import extract_loudness, extract_pitch
import librosa
import yaml 
from nnAudio import Spectrogram

with open("syntheon/inferencer/vital/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
sr = config["common"]["sampling_rate"]
n_mfcc = config["train"]["n_mfcc"]

spec = Spectrogram.MFCC(sr=sr, n_mfcc=n_mfcc)


def sanitize_onsets(times, onset_frames, onset_strengths):
    """
    times: actual timestamp per frame in STFT (by hop length)
            e.g. [0, 0.3, 0.6, ...]
    onset_frames: index list, index on `times` to get an onset event
    onset_strengths: get strength per frame. same shape as times.
                        so when strength is high will have a onset event, index in `onset_frames`  
    """
    # TODO: need to check if we need this always
    res_frames = []

    cur_frame = onset_frames[0]
    cur_time = times[cur_frame]
    res_frames.append(cur_frame)

    for frame in onset_frames[1:]:
        if times[frame] - cur_time > 0.05:  # TODO: parameterize
            res_frames.append(frame)
        cur_frame = frame
        cur_time = times[frame]

    return np.array(res_frames)


def aggregate(vals):
    """
    aggregate the window of pitch values.
    rationale: bin pitch values (to reduce fluctuation), get the bin with most values within the window
    """
    bins = {}
    for val in vals:
        bin = val // 10
        if bin in bins:
            bins[bin].append(val)
        else:
            bins[bin] = [val]
    
    sorted_bins = sorted(bins.keys())
    max_len_bin = sorted_bins[0]

    for bin in sorted_bins:
        if len(bins[bin]) > len(bins[max_len_bin]):
            max_len_bin = bin
    
    return bins[max_len_bin][0]



def monotonize_pitch(times, onset_frames, pitch):
    """
    remove wobbling frequencies in pitch. take the pitch value on the onset frame
    problem is accuracy issue -- need to align onset and pitch
    because librosa onset might read wrong pitch from crepe output
    """
    res_pitch = np.zeros(pitch.shape)
    pitch_map_lst = []

    prev_ts = times[onset_frames[0]]

    for idx, frame in enumerate(onset_frames):
        if idx == 0:
            continue
        ts = times[frame]
        pitch_vals = pitch[int(prev_ts * 100) : int(ts * 100)]

        if len(pitch_vals) > 0:
            cur_pitch = aggregate(pitch_vals)
            pitch_map_lst.append((int(prev_ts * 100), cur_pitch))
            prev_ts = ts

    # for final frame
    ts = times[-1]
    pitch_vals = pitch[int(prev_ts * 100) : int(ts * 100)]
    if len(pitch_vals) > 0:
        cur_pitch = aggregate(pitch_vals)
        pitch_map_lst.append((int(prev_ts * 100), cur_pitch))
    
    if pitch_map_lst[0][0] == 0:
        res_pitch[0] = pitch_map_lst[0][1]
        cur_pitch = pitch_map_lst[0][1]
        cur_idx = 1
    else:
        res_pitch[0] = 0
        cur_pitch = 0
        cur_idx = 0
    
    for i in range(1, len(pitch)):
        if i == pitch_map_lst[cur_idx][0]:
            cur_pitch = pitch_map_lst[cur_idx][1]
            res_pitch[i] = cur_pitch
            if cur_idx < len(pitch_map_lst) - 1:
                cur_idx += 1
        else:
            res_pitch[i] = cur_pitch

    return res_pitch
    

def preprocess(f, sampling_rate, block_size, signal_length=-1, oneshot=True):
    x, sr = librosa.load(f, sampling_rate)
    if signal_length == -1:     # full length
        signal_length = len(x)
    else:
        if len(x) > signal_length:
            x = x[:signal_length*sampling_rate]
        elif len(x) < signal_length:
            N = (signal_length - len(x) % signal_length) % signal_length
            x = np.pad(x, (0, N))

        if oneshot:
            x = x[..., :signal_length]

    D = np.abs(librosa.stft(x))
    times = librosa.times_like(D, sr=sr)
    onset_strengths = librosa.onset.onset_strength(y=x, sr=sr, aggregate=np.median)
    onset_frames = librosa.onset.onset_detect(y=x, sr=sr)

    onset_frames = sanitize_onsets(times, onset_frames, onset_strengths)

    # TODO: HACK for now, onset detector missed. not all samples need this!!
    onset_frames = np.concatenate([np.array([0]), onset_frames])

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    pitch_monotonize = monotonize_pitch(times, onset_frames, pitch)
    pitch = pitch_monotonize
    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1).squeeze()
    loudness = loudness.reshape(x.shape[0], -1)

    # prepare for inference input
    x = torch.tensor(x)
    pitch = torch.tensor(pitch).unsqueeze(0)
    loudness = torch.tensor(loudness)

    x = torch.cat([x, x], dim=0)
    pitch = torch.cat([pitch, pitch], dim=0)
    loudness = torch.cat([loudness, loudness], dim=0)

    mean_loudness, std_loudness = -39.74668743704927, 54.19612404969509
    pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
    loudness = (loudness - mean_loudness) / std_loudness

    mfcc = spec(x)

    return x, pitch, loudness, times, onset_frames, mfcc
