"""
Differentiable wavetable synthesis component.
"""
import torch
from torch import nn
import numpy as np
from syntheon.inferencer.vital.models.utils import *
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
from syntheon.inferencer.vital.models.core import upsample
import random
import time
from syntheon.inferencer.vital.models.adsr_envelope import *


def wavetable_osc(wavetable, freq, sr):
    """
    General wavetable synthesis oscilator.
    wavetable: (wavetable_len,)
    freq: (batch_size, dur * sr)
    sr: const
    """
    freq = freq.squeeze()
    increment = freq / sr * wavetable.shape[0]
    index = torch.cumsum(increment, dim=1) - increment[0]
    index = index % wavetable.shape[0]

    # uses linear interpolation implementation
    index_low = torch.floor(index.clone())
    index_high = torch.ceil(index.clone())
    alpha = index - index_low
    index_low = index_low.long()
    index_high = index_high.long()

    output = wavetable[index_low] + alpha * (wavetable[index_high % wavetable.shape[0]] - wavetable[index_low])
        
    return output


def wavetable_osc_v2(wavetable, freq, sr):
    """
    General wavetable synthesis oscilator, wavetable per item in batch
    wavetable: (batch_size, wavetable_len,)
    freq: (batch_size, dur * sr)
    sr: const
    """
    freq = freq.squeeze()
    increment = freq / sr * wavetable.shape[1]
    index = torch.cumsum(increment, dim=1) - increment[1]
    index = index % wavetable.shape[1]

    # uses linear interpolation implementation
    index_low = torch.floor(index.clone())
    index_high = torch.ceil(index.clone())
    alpha = index - index_low
    index_low = index_low.long()
    index_high = index_high.long()

    batch_size = wavetable.shape[0]
    output = []

    # TODO: do for loop for now, think any ways to parallelize this (einsum?)
    for bs in range(batch_size):
        wt, idx_l, idx_h, alp = wavetable[bs], index_low[bs].unsqueeze(0), index_high[bs].unsqueeze(0), alpha[bs].unsqueeze(0)
        signal = wt[idx_l] + alp * (wt[idx_h % wt.shape[0]] - wt[idx_l])
        output.append(signal)
    
    output = torch.cat(output, dim=0)
    return output


def generate_wavetable(length, f, cycle=1, phase=0):
    """
    Generate a wavetable of specified length using 
    function f(x) where x is phase.
    Period of f is assumed to be 2 pi.
    """
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(cycle * 2 * np.pi * i / length + 2 * phase * np.pi)
    return torch.tensor(wavetable)


class WavetableSynth(nn.Module):
    def __init__(self,
                 wavetables=None,
                 n_wavetables=64,
                 wavetable_len=512,
                 sr=44100,
                 duration_secs=3,
                 block_size=160,
                 is_initial_wt_trainable=True):
        super(WavetableSynth, self).__init__()
        if wavetables is None: 
            self.wavetables = []
            for _ in range(n_wavetables):
                cur = nn.Parameter(torch.empty(wavetable_len).normal_(mean=0, std=0.01))
                self.wavetables.append(cur)

            self.wavetables = nn.ParameterList(self.wavetables)

            for idx, wt in enumerate(self.wavetables):
                # following the paper, initialize f0-f3 wavetables and disable backprop
                # if idx == 0:
                #     wt.data = generate_wavetable(wavetable_len, np.sin, cycle=2, phase=random.uniform(0, 1))
                #     wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                #     wt.requires_grad = is_initial_wt_trainable
                # else:
                wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                wt.requires_grad = is_initial_wt_trainable
        else:
            self.wavetables = wavetables
        
        self.attention = nn.Parameter(torch.ones(n_wavetables,).cuda())
        self.sr = sr
        self.block_size = block_size
        self.attention_softmax = nn.Softmax(dim=0)
        self.duration_secs = duration_secs

    def forward(self, pitch, amplitude):
        output_waveform_lst = []
        for wt_idx in range(len(self.wavetables)):
            wt = self.wavetables[wt_idx]
            if wt_idx not in [0, 1, 2, 3]:
                wt = nn.Tanh()(wt)  # ensure wavetable range is between [-1, 1]
            waveform = wavetable_osc(wt, pitch, self.sr)
            output_waveform_lst.append(waveform)

        # apply attention 
        attention = self.attention_softmax(self.attention)
        attention_output = attention
        attention = torch.stack(100 * self.duration_secs * [attention], dim=-1)
        attention_upsample = upsample(attention.unsqueeze(-1), self.block_size).squeeze()

        output_waveform = torch.stack(output_waveform_lst, dim=1)
        output_waveform = output_waveform * attention_upsample
        output_waveform_after = torch.sum(output_waveform, dim=1)
      
        output_waveform_after = output_waveform_after.unsqueeze(-1)
        output_waveform_after = output_waveform_after * amplitude
       
        return output_waveform_after, attention_output


class WavetableSynthV2(nn.Module):
    """
    take wavetable as input, not model parameters
    """
    def __init__(self,
                 sr=44100,
                 duration_secs=4,
                 block_size=160,
                 enable_amplitude=True):
        """
        Turn on smoothing to reduce noise in learnt wavetables.
        Smoothing takes in a 0-1 value, which is window size ratio w.r.t. wavetable length
        Also a max_smooth_window_size is specified
        """
        super(WavetableSynthV2, self).__init__()       
        self.sr = sr
        self.block_size = block_size
        self.duration_secs = duration_secs
        self.enable_amplitude = enable_amplitude

    def forward(self, pitch, amplitude, wavetables, attention):
        """
        batch size version
        input:
        wavetables: (bs, n_wavetables, wavetable_len), -1 to 1
        attention: softmax-ed, (bs, n_wavetables,)
        smoothing_coeff: (bs, ), 0 to 1

        output:
        (bs, dur * sr)
        """
        output_waveform_lst = []
        for wt_idx in range(wavetables.shape[1]):
            wt = wavetables[:, wt_idx, :]
            waveform = wavetable_osc_v2(wt, pitch, self.sr)

            output_waveform_lst.append(waveform)

        # apply attention 
        attention_upsample = torch.stack(100 * self.duration_secs * [attention], dim=-1)
        attention_upsample = upsample(torch.permute(attention_upsample, (1, 2, 0)), self.block_size)
        if (attention_upsample.shape[0] != 1):
            attention_upsample = attention_upsample.squeeze()   # TODO: a little hacky code here, need to remove
        attention_upsample = torch.permute(attention_upsample, (2, 0, 1))

        output_waveform = torch.stack(output_waveform_lst, dim=1)
        output_waveform = output_waveform * attention_upsample
        output_waveform_after = torch.sum(output_waveform, dim=1)
      
        output_waveform_after = output_waveform_after.unsqueeze(-1)
        if self.enable_amplitude:
            output_waveform_after = output_waveform_after * amplitude
       
        return output_waveform_after, attention


if __name__ == "__main__":
    # create a sine wavetable and to a simple synthesis test
    wavetable_len = 512
    sr = 16000
    duration = 4
    freq_t_1 = [739.99 for _ in range(sr)] + [523.25 for _ in range(sr)] + [349.23 for _ in range(sr * 2)]
    freq_t_1 = torch.tensor(freq_t_1)
    freq_t_2 = [523.25 for _ in range(sr)] + [349.23 for _ in range(sr)] + [739.99 for _ in range(sr * 2)]
    freq_t_2 = torch.tensor(freq_t_2)
    freq_t_3 = [349.23 for _ in range(sr)] + [739.99 for _ in range(sr)] + [523.25 for _ in range(sr * 2)]
    freq_t_3 = torch.tensor(freq_t_3)

    pitch, onset_frames, times = np.load("pitch.npy"), np.load("onset.npy"), np.load("times.npy")
    pitch = torch.tensor(pitch)
    pitch = upsample(pitch.unsqueeze(-1).unsqueeze(0), 160).squeeze()

    freq_t = torch.stack([pitch, pitch, pitch], dim=0)
    sine_wavetable = generate_wavetable(wavetable_len, np.sin)
    from utils import sawtooth_waveform
    saw_wavetable = generate_wavetable(wavetable_len, sawtooth_waveform)
    square_wavetable = generate_wavetable(wavetable_len, square_waveform)

    wavetable = torch.stack([sine_wavetable, saw_wavetable, square_wavetable], dim=0)

    # test batch wavetable_osc
    signal = wavetable_osc_v2(wavetable, freq_t, sr)

    # test with adsr
    shaper = ADSREnvelopeShaper()
    adsr = get_amp_shaper_v2(shaper, times[onset_frames], 
                            attack_secs=torch.tensor([0.00]),
                            decay_secs=torch.tensor([0.05]),
                            sustain_level=torch.tensor([0.0]))
    if adsr.shape[0] < 400:
        append_tensor = torch.tensor([adsr[-1]] * (400 - adsr.shape[0]))
        adsr = torch.cat([adsr, append_tensor], dim=-1)
    else:
        adsr = adsr[:400]
    adsr = upsample(adsr.unsqueeze(-1).unsqueeze(0), 160).squeeze()

    signal = signal * adsr
    
    # wt_synth = WavetableSynth(wavetables=sine_wavetable, sr=sr, duration_secs=4)
    # amplitude_t = torch.ones(sr * duration,)
    # amplitude_t = torch.stack([amplitude_t, amplitude_t, amplitude_t], dim=0)
    # amplitude_t = amplitude_t.unsqueeze(-1)

    # y = wt_synth(freq_t, amplitude_t, duration)
    # print(y.shape, 'y')
    # plt.plot(y.squeeze()[0].detach().numpy())
    # plt.show()
    sf.write('test_3s_v1.wav', signal.squeeze()[0].detach().numpy(), sr, 'PCM_24')
    sf.write('test_3s_v2.wav', signal.squeeze()[1].detach().numpy(), sr, 'PCM_24')
    sf.write('test_3s_v3.wav', signal.squeeze()[2].detach().numpy(), sr, 'PCM_24')