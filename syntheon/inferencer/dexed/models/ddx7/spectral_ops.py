import torch
import torch.nn.functional as F
import torchcrepe
import torchaudio
import librosa
from syntheon.inferencer.dexed.models.ddx7.core import _DB_RANGE,_REF_DB
import math
import numpy as np
from time import time

_RMS_FRAME = 2048
_CREPE_WIN_LEN = 1024
_LD_N_FFT = 2048

def safe_log(x):
    return torch.log(x + 1e-7)

def calc_f0(audio, rate, hop_size,fmin,fmax,model,
            batch_size,device,center=False):
    if center is False:
      # Add padding to the end. Then execute crepe w/o padding.
      # Crepe pads so that the signal stays in the center.
      n_samples_initial = int(audio.shape[-1])
      n_frames = int(np.ceil(n_samples_initial / hop_size))
      n_samples_final = (n_frames - 1) * hop_size + _CREPE_WIN_LEN
      pad = n_samples_final - n_samples_initial
      audio = np.pad(audio, ((0, pad),), "constant")

    audio = torch.from_numpy(audio).unsqueeze(0).float().to(device)

    t1 = time()
    print("predicting...")
    crepe_tuple = torchcrepe.predict(audio,
                        rate,
                        hop_size,
                        fmin,
                        fmax,
                        model,
                        return_periodicity=True,
                        batch_size=batch_size,
                        device=device,
                        pad=center)
    print("done...", time() - t1)

    f0 = crepe_tuple[0]
    confidence = crepe_tuple[1]
    if center is True:
      f0 = f0[:,0:-1] #Discard the last sample
      confidence = confidence[:,0:-1] #Discard the last sample

    f0 = f0.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()
    return f0,confidence

def calc_loudness(audio, rate, n_fft=_LD_N_FFT, hop_size=64,
                  range_db=_DB_RANGE,ref_db=_REF_DB,center=False):
    np.seterr(divide='ignore')

    """Compute loudness, add to example (ref is white noise, amplitude=1)."""
    # Copied from magenta/ddsp/spectral_ops.py
    # Get magnitudes.
    if center is False:
        # Add padding to the end
        n_samples_initial = int(audio.shape[-1])
        n_frames = int(np.ceil(n_samples_initial / hop_size))
        n_samples_final = (n_frames - 1) * hop_size + n_fft
        pad = n_samples_final - n_samples_initial
        audio = np.pad(audio, ((0, pad),), "constant")
    spectra = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_size, center=center).T

    # Compute power
    amplitude = np.abs(spectra)
    amin = 1e-20  # Avoid log(0) instabilities.
    power_db = np.log10(np.maximum(amin, amplitude))
    power_db *= 20.0

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[np.newaxis, :]
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= ref_db
    loudness = np.maximum(loudness, -range_db)

    # Average over frequency bins. (loudness is taken from the fft dimension!)
    mean_loudness_db = np.mean(loudness, axis=-1)
    return mean_loudness_db.astype(np.float32)

'''
RMS POWER COMPUTATION.
'''

def amplitude_to_db(amplitude):
  """Converts amplitude to decibels."""
  amin = 1e-20  # Avoid log(0) instabilities.
  db = np.log10(np.maximum(amin, amplitude))
  db *= 20.0
  return db

def compute_rms_energy(audio,
                       frame_size=2048,
                       hop_size=64,
                       pad_end=True):
  """Compute root mean squared energy of audio."""
  if pad_end is True:
    # Add padding to the end
    n_samples_initial = int(audio.shape[-1])
    n_frames = int(np.ceil(n_samples_initial / hop_size))
    n_samples_final = (n_frames - 1) * hop_size + frame_size
    pad = n_samples_final - n_samples_initial
    audio = np.pad(audio, ((0, pad),), "constant")

  audio = torch.tensor(audio)
  audio_frames = audio.unfold(-1,frame_size,hop_size)
  rms_energy = torch.mean(audio_frames**2.0,dim=-1)**0.5

  return rms_energy.cpu().numpy()


def calc_power(audio,
                frame_size=_RMS_FRAME,
                hop_size=64,
                range_db=_DB_RANGE,
                ref_db=20.7,
                pad_end=True):
  """Compute power of audio in dB."""
  rms_energy = compute_rms_energy(audio, frame_size, hop_size,pad_end=pad_end)
  power_db = amplitude_to_db(rms_energy**2)
  #print(power_db)
  # Set dynamic range.
  power_db -= ref_db
  power_db = np.maximum(power_db, -range_db)
  return power_db.astype(np.float32)
