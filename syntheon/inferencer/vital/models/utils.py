import numpy as np
import librosa
import soundfile as sf


# helper functions to generate wavetable
def generate_wavetable(length, f):
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(2 * np.pi * i / length)
    return wavetable


def sawtooth_waveform(x):
    """Sawtooth with period 2 pi."""
    return (x + np.pi) / np.pi % 2 - 1


def square_waveform(x):
    """Square waveform with period 2 pi."""
    return np.sign(np.sin(x))


def trim_audio(in_name, out_name, start_sec, end_sec, sr=44100):
    x, sr = librosa.load(in_name, sr=sr)
    x = x[start_sec * sr: end_sec * sr]
    sf.write(out_name, x, sr, 'PCM_24')


if __name__ == "__main__":
    trim_audio("test_audio/kygo_pluck.mp3", "kygo_pluck.wav", 75, 85)