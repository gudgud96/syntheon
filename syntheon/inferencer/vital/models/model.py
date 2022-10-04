"""
Diff-WTS model. Main adapted from https://github.com/acids-ircam/ddsp_pytorch.
"""
from syntheon.inferencer.vital.models.core import harmonic_synth
from syntheon.inferencer.vital.models.wavetable_synth import WavetableSynth
import torch
import torch.nn as nn
from syntheon.inferencer.vital.models.core import mlp, gru, scale_function, remove_above_nyquist, upsample
from syntheon.inferencer.vital.models.core import amp_to_impulse_response, fft_convolve
from syntheon.inferencer.vital.models.adsr_envelope import *
import numpy as np
from torchvision.transforms import Resize
import matplotlib.pyplot as plt


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class WTS(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size, n_wavetables, mode="wavetable", duration_secs=3,
                 adsr_hidden_size=1200,
                ):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = mlp(30, hidden_size, 3)
        self.layer_norm = nn.LayerNorm(30)
        self.gru_mfcc = nn.GRU(30, 512, batch_first=True)
        self.mlp_mfcc = nn.Linear(512, 16)

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3),
                                      mlp(1, hidden_size, 3),
                                      mlp(16, hidden_size, 3)])
        self.gru = gru(3, hidden_size)
        self.out_mlp = mlp(hidden_size * 4, hidden_size, 3)

        self.loudness_mlp = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        # for adsr learning
        self.adsr_conv1d = nn.Conv1d(1, 1, block_size, stride=block_size)
        self.attack_sec_head = nn.Sequential(
            nn.Linear(300, 1),
            nn.ReLU()
        )
        self.decay_sec_head = nn.Sequential(
            nn.Linear(300, 1),
            nn.ReLU()
        )
        self.sustain_level_head = nn.Sequential(
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

        self.reverb = Reverb(sampling_rate, sampling_rate)
        self.wts = WavetableSynth(n_wavetables=n_wavetables, 
                                  sr=sampling_rate, 
                                  duration_secs=duration_secs,
                                  block_size=block_size)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.mode = mode
        self.duration_secs = duration_secs
        self.adsr = None

        self.shaper = ADSREnvelopeShaper()

    def forward(self, y, mfcc, pitch, loudness, times, onset_frames):
        # encode mfcc first
        # use layer norm instead of trainable norm, not much difference found
        mfcc = self.layer_norm(torch.transpose(mfcc, 1, 2))
        mfcc = self.gru_mfcc(mfcc)[0]
        mfcc = self.mlp_mfcc(mfcc)

        # use image resize to align dimensions, ddsp also do this...
        mfcc = Resize(size=(self.duration_secs * 100, 16))(mfcc)

        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
            self.in_mlps[2](mfcc)
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], hidden], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = self.proj_matrices[0](hidden)
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        total_amp_2 = self.loudness_mlp(loudness)

        amplitudes = upsample(amplitudes, self.block_size)
        pitch_prev = pitch
        pitch = upsample(pitch, self.block_size)
        total_amp = upsample(total_amp, self.block_size)    # TODO: wts can't backprop when using this total_amp, not sure why
        total_amp_2 = upsample(total_amp_2, self.block_size)    # use this instead for wavetable

        # diff-wave-synth synthesizer
        harmonic, attention_output = self.wts(pitch, total_amp_2)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        # adsr shaping
        y_hidden = self.adsr_conv1d(y.unsqueeze(1))
        attack_secs = self.attack_sec_head(y_hidden.squeeze()).squeeze()
        decay_secs = self.decay_sec_head(y_hidden.squeeze()).squeeze()
        sustain_level = self.sustain_level_head(y_hidden.squeeze()).squeeze()

        amp_onsets = np.append(times[onset_frames], np.array([times[-1]]))

        adsr = get_amp_shaper(self.shaper, amp_onsets, 
                            attack_secs=attack_secs,
                            decay_secs=decay_secs,
                            sustain_level=sustain_level)
        if adsr.shape[0] < pitch_prev.shape[1]:
            adsr = torch.nn.functional.pad(adsr, (0, pitch_prev.shape[1] - adsr.shape[0]), "constant", adsr[-1].item())
        else:
            adsr = adsr[:pitch_prev.shape[1]]
        
        self.adsr = adsr

        adsr = adsr.unsqueeze(0).unsqueeze(-1)
        adsr = upsample(adsr, self.block_size).squeeze().unsqueeze(-1)

        # temporary: fix adsr for more examples
        adsr = torch.stack([adsr] * signal.shape[0], dim=0)
        final_signal = signal * adsr
        # final_signal = signal

        # reverb part
        # signal = self.reverb(signal)

        return signal, adsr, final_signal, attention_output