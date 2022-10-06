"""
Diff-WTS model. Main adapted from https://github.com/acids-ircam/ddsp_pytorch.
"""
from syntheon.inferencer.vital.models.core import harmonic_synth
from syntheon.inferencer.vital.models.wavetable_synth import WavetableSynthV2
import torch
import torch.nn as nn
from syntheon.inferencer.vital.models.core import mlp, gru, scale_function, remove_above_nyquist, upsample
from syntheon.inferencer.vital.models.core import amp_to_impulse_response, fft_convolve
from syntheon.inferencer.vital.models.adsr_envelope import *
import numpy as np
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from time import time

class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(self.name, x[0].squeeze().item())
        x += 1e-2
        return x


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


def infer_wavetables(y, pitch):
    """
    TODO: do for single sample first, need to batch it later
    TODO: VERY BUGGY CODE. didn't care for edge cases (like silence). need to add the checks in

    y: (64000,)
    pitch: (400,), 1 second 100 frames
    """
    period = 1 / pitch * 16000
    
    # find the first continuous pitch
    # TODO: find the most continuous pitch across the sample, for best results
    continuous_threshold = 10   # at least 10 steps = 0.1 sec, will be a problem for plucks, but for now do it like this
    continuous_pitch = -1
    continuous_pitch_idx = -1
    cur_pitch = pitch[0]
    step = 0
    
    t1 = time()
    for idx in range(1, len(pitch)):
        if abs(pitch[idx] - cur_pitch) < 1e-2: # equal freq tolerance 1e-2
            step += 1
            if step > continuous_threshold:
                continuous_pitch = cur_pitch
                continuous_pitch_idx = idx - step
                break
        else:
            cur_pitch = pitch[idx]
            step = 0
    
    if continuous_pitch == -1:  # fallback
        continuous_pitch = pitch[0]
        continuous_pitch_idx = 0
    
    period = int(1 / continuous_pitch * 16000)
    pitch_offset_idx = continuous_pitch_idx * 160  # 160 = sr / frame_size (100)
        
    # find local minimum within a window of 2 periods
    cur = y[pitch_offset_idx : pitch_offset_idx + 1600]
    min_idx = torch.argmin(cur).item()
    
    # here we take first wavelet, but also can take the average of a few wavelets
    # TODO: prone to silence right now. need to fix. now HACK search for local minima across 1600 samples to solve
    wavelet = y[min_idx : min_idx + period]
    
    # upsample + normalize magnitude
    wavelet_tensor = wavelet.clone().detach().unsqueeze(-1).unsqueeze(0)
    if torch.isinf(wavelet_tensor).any() or torch.isnan(wavelet_tensor).any():
        print('wavelet tensor has inf or nan', torch.isinf(wavelet_tensor).any(), torch.isnan(wavelet_tensor).any())
    wavelet_upsample = upsample(wavelet_tensor, factor=0, preferred_size=512, mode="linear").squeeze()
    if torch.isinf(wavelet_upsample).any() or torch.isnan(wavelet_upsample).any():
        print('wavelet upsample has inf or nan', torch.isinf(wavelet_upsample).any(), torch.isnan(wavelet_upsample).any())
    if wavelet_upsample.max() - wavelet_upsample.min() < 1e-4:
        # don't min-max norm in this case
        pass
    else:
        wavelet_upsample = (wavelet_upsample - wavelet_upsample.min()) / \
                            (wavelet_upsample.max() - wavelet_upsample.min())
        wavelet_upsample = wavelet_upsample * 2 - 1
    if torch.isinf(wavelet_upsample).any() or torch.isnan(wavelet_upsample).any():
        print('wavelet upsample 2 has inf or nan', torch.isinf(wavelet_upsample).any(), torch.isnan(wavelet_upsample).any(),
                wavelet_upsample.max(), wavelet_upsample.min())

    return wavelet_upsample


class WTSv2(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size, mode="wavetable", duration_secs=3, num_wavetables=3,
                 wavetable_smoothing=False, min_smoothing_sigma=0.5, max_smoothing_sigma=50,
                 preload_wt=False, is_round_secs=False, enable_amplitude=True, device='cuda'
                ):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # feature extractors
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

        # for wavetable learning
        self.wt1_conv1d = nn.Sequential(
            nn.Conv1d(1, num_wavetables, 16, stride=16),     # 3 here is num_wavetable
            nn.Tanh(),
            nn.Conv1d(num_wavetables, num_wavetables, 8, stride=8),
            nn.Tanh(),
            nn.Linear(500, 512),                # 512 is wavetable length
            nn.Tanh()
        )
        self.attention_wt1 = nn.Linear(512, 1)
        self.smoothing_linear = nn.Linear(512, 1)
        self.smoothing_sigmoid = nn.Sigmoid()

        # for adsr learning
        self.shaper = ADSREnvelopeShaper(is_round_secs)
        self.adsr_conv1d = nn.Conv1d(1, 1, block_size, stride=block_size)

        self.attack_gru = nn.GRU(1, 8, batch_first=True, bidirectional=True)
        self.decay_gru = nn.GRU(1, 8, batch_first=True, bidirectional=True)
        self.sustain_gru = nn.GRU(1, 8, batch_first=True, bidirectional=True)

        self.attack_sec_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.decay_sec_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.sustain_level_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # for adsr result storage
        self.attack_sec = nn.Parameter(torch.ones(1,))
        self.decay_sec = nn.Parameter(torch.ones(1,))
        self.sustain_level = nn.Parameter(torch.ones(1,))       

        self.max_attack_secs = 2.0
        self.max_decay_secs = 2.0 

        # for synthesis
        self.reverb = Reverb(sampling_rate, sampling_rate)
        self.wts = WavetableSynthV2(sr=sampling_rate, 
                                    duration_secs=duration_secs,
                                    block_size=block_size,
                                    enable_amplitude=enable_amplitude)
        self.wavetable_smoothing = wavetable_smoothing
        self.min_smoothing_sigma = min_smoothing_sigma
        self.max_smoothing_sigma = max_smoothing_sigma

        self.preload_wt = preload_wt

        self.mode = mode
        self.duration_secs = duration_secs
        self.device = device

    def forward(self, y, mfcc, pitch, loudness, times, onset_frames):
        batch_size = y.shape[0]

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
        total_amp = self.loudness_mlp(loudness)
        pitch_prev = pitch

        # TODO: upsample is very slow
        pitch = upsample(pitch, self.block_size)
        total_amp = upsample(total_amp, self.block_size)    # use this instead for wavetable

        # diff-wave-synth synthesizer
        if self.preload_wt:
            # TODO: very slow implementation...
            wavetables = []
            for idx in range(batch_size):
                wt = infer_wavetables(y[idx].squeeze(), pitch_prev[idx].squeeze())
                wavetables.append(wt)
            wavetables = torch.stack(wavetables, dim=0).unsqueeze(1)
            if torch.isinf(wavetables).any() or torch.isnan(wavetables).any():
                print('wavetables has inf or nan', torch.isinf(wavetables).any(), torch.isnan(wavetables).any())
        else:
            wavetables = self.wt1_conv1d(y.unsqueeze(1))

        if self.wavetable_smoothing:
            smoothing_coeff = self.smoothing_linear(wavetables)
            smoothing_coeff = smoothing_coeff.squeeze(1)        # HACK: here should assume only 1 wavetable
            smoothing_coeff = self.smoothing_sigmoid(smoothing_coeff)
            wavetables_old = wavetables
            wavetables = self.smoothing(wavetables, smoothing_coeff)
        else:
            wavetables_old = None
            smoothing_coeff = None

        attention_output = self.attention_wt1(wavetables).squeeze(-1)
        attention_output = nn.Softmax(dim=-1)(attention_output)

        harmonic, attention_output = self.wts(pitch, total_amp, wavetables, attention_output)

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
        output_attack, hn_attack = self.attack_gru(loudness)
        hn_attack = torch.cat([hn_attack[0], hn_attack[1]], dim=-1)
        output_decay, hn_decay = self.decay_gru(loudness)
        hn_decay = torch.cat([hn_decay[0], hn_decay[1]], dim=-1)
        output_sustain, hn_sustain = self.sustain_gru(loudness)
        hn_sustain = torch.cat([hn_sustain[0], hn_sustain[1]], dim=-1)

        # print(hn_decay[:10])
        attack_level = self.attack_sec_head(hn_attack).squeeze()            # 0-1
        decay_level = self.decay_sec_head(hn_decay).squeeze()               # 0-1
        sustain_level = self.sustain_level_head(hn_sustain).squeeze()

        attack_secs = attack_level * self.max_attack_secs
        decay_secs = decay_level * self.max_decay_secs
        # print(attack_secs, decay_secs, sustain_level)
        # attack_secs = torch.tensor([0.0, 0.0]).cuda()
        # decay_secs = torch.tensor([0.1, 0.1]).cuda()
        # sustain_level = torch.tensor([0.001, 0.001]).cuda()

        amp_onsets = np.append(times[onset_frames], np.array([times[-1]]))  # TODO: now 1 onset is enough, because all training samples pitch are the same

        adsr = get_amp_shaper_v2(self.shaper, amp_onsets, 
                                attack_secs=attack_secs,
                                decay_secs=decay_secs,
                                sustain_level=sustain_level)
        if adsr.shape[1] < pitch_prev.shape[1]:
            # adsr = torch.nn.functional.pad(adsr, (0, pitch_prev.shape[1] - adsr.shape[1]), "constant", adsr[-1].item())
            adsr = torch.cat([adsr, adsr[:, -1].unsqueeze(-1)], dim=-1)
        else:
            adsr = adsr[:pitch_prev.shape[1]]
        
        self.adsr = adsr

        adsr_prev = adsr
        adsr = adsr.unsqueeze(-1)
        adsr = upsample(adsr, self.block_size).squeeze(-1)   

        adsr = adsr[:, :signal.shape[1]]

        final_signal = signal.squeeze() * adsr

        # final_signal = signal

        # reverb part
        # signal = self.reverb(signal)

        return signal, (attack_secs, decay_secs, sustain_level), final_signal, attention_output, wavetables, wavetables_old, smoothing_coeff
        # return signal, adsr_prev, final_signal, attention_output, wavetables, wavetables_old, smoothing_coeff
    
    def smoothing_deprecate(self, wavetables, p):
        """
        wavetables: size (b, wavetable_length)
        p: size (b,). value between 0-1
        """
        wavetables = wavetables.squeeze()   # HACK: also a hack, because by right wavetables.dim[1] is num_wavetables, here assume 1 wavetable
        bs, wavetable_length = wavetables.shape[0], wavetables.shape[1]
        smoothed_wavetables = torch.zeros((bs, wavetable_length))
        if self.device == "cuda":
            smoothed_wavetables = smoothed_wavetables.cuda()

        sigma = p * (self.max_smoothing_sigma - self.min_smoothing_sigma) + self.min_smoothing_sigma

        x_vals = torch.arange(wavetable_length)
        if self.device == "cuda":
            x_vals = x_vals.cuda()
        x_vals = torch.stack([x_vals] * bs, dim=0)

        # TODO: can we not do for loop here?
        for x_position in range(wavetable_length):
            kernel = torch.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
            kernel = kernel / torch.sum(kernel, dim=1).unsqueeze(-1)
            smoothed_wavetables[:, x_position] = torch.sum(wavetables * kernel, dim=-1)
    
        return smoothed_wavetables.unsqueeze(1)
    
    def smoothing(self, wavetables, p):       
        bs, wavetable_length = wavetables.shape[0], wavetables.shape[2]
        smoothed_wavetables = torch.zeros((bs, wavetable_length))
        if self.device == "cuda":
            smoothed_wavetables = smoothed_wavetables.cuda()
        
        sigma = p * (self.max_smoothing_sigma - self.min_smoothing_sigma) + self.min_smoothing_sigma
        sigma = sigma.unsqueeze(-1)                         # size (bs, 1, 1)
        
        kernel = torch.arange(wavetable_length)
        if self.device == "cuda":
            kernel = kernel.cuda()
        kernel = kernel.unsqueeze(0) - kernel.unsqueeze(-1)  # x_position - x_vals, size (wt_len, wt_len)

        kernel = torch.exp(-kernel ** 2 / (2 * sigma ** 2))   # size (b, wt_len, wt_len)    
        kernel = kernel / torch.sum(kernel, dim=-1).unsqueeze(-1)          # dim 1 or dim -1?
            
        # wavetables = wavetables.unsqueeze(1)
        smoothed_wavetables = torch.bmm(wavetables, kernel)  # (bs, 1, wt_len) * (bs, wt_len, wt_len)
        return smoothed_wavetables