import torch
import torch.nn as nn
import math
from syntheon.inferencer.dexed.models.ddx7.core import *
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

def exp_sigmoid(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


class FMSynth(nn.Module):
    def __init__(self,sample_rate,block_size,fr=[1,1,1,1,3,14],max_ol=2,
        scale_fn = torch.sigmoid,synth_module='fmstrings',is_reverb=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reverb = Reverb(length=sample_rate, sample_rate=sample_rate)
        fr = torch.tensor(fr) # Frequency Ratio
        self.register_buffer("fr", fr) #Non learnable but sent to GPU if declared as buffers, and stored in model dictionary
        self.scale_fn = scale_fn
        self.use_cumsum_nd = False
        self.max_ol = max_ol
        self.is_reverb = is_reverb

        available_synths = {
            'fmbrass': fm_brass_synth,
            'fmflute': fm_flute_synth,
            'fmstrings': fm_string_synth,
            'fmablbrass': fm_ablbrass_synth,
            '2stack2': fm_2stack2,
            '1stack2':fm_1stack2,
            '1stack4': fm_1stack4}

        self.synth_module = available_synths[synth_module]

    def forward(self,controls):

        ol = self.max_ol*self.scale_fn(controls['ol'])
        ol_up = upsample(ol, self.block_size,'linear')
        f0_up = upsample(controls['f0_hz'], self.block_size,'linear')
        signal = self.synth_module(f0_up,
                                ol_up,
                                self.fr,
                                self.sample_rate,
                                self.max_ol,
                                self.use_cumsum_nd)
        #reverb part
        if self.is_reverb:
            signal = self.reverb(signal)

        synth_out = {
            'synth_audio': signal,
            'ol': ol,
            'f0_hz': controls['f0_hz']
            }
        return synth_out

class HNSynth(nn.Module):
    def __init__(self,sample_rate,block_size,scale_fn = exp_sigmoid):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reverb = Reverb(length=sample_rate, sample_rate=sample_rate)
        self.use_cumsum_nd = False
        self.scale_fn = scale_fn

    # expects: harmonic_distr, amplitude, noise_bands
    def forward(self,controls):

        harmonics = self.scale_fn(controls['harmonic_distribution'])
        noise_bands = self.scale_fn(controls['noise_bands'])
        total_amp = self.scale_fn(controls['amplitude'])

        harmonics = remove_above_nyquist(
            harmonics,
            controls['f0_hz'],
            self.sample_rate,
        )
        harmonics /= harmonics.sum(-1, keepdim=True)
        harmonics *= total_amp

        harmonics_up = upsample(harmonics, self.block_size)
        f0_up = upsample(controls['f0_hz'], self.block_size,'linear')

        harmonic = harmonic_synth(f0_up, harmonics_up, self.sample_rate, self.use_cumsum_nd)
        impulse = amp_to_impulse_response(noise_bands, self.block_size)

        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
            ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)
        synth_out = {
            'synth_audio': signal,
            'harmonic_distribution': harmonics,
            'noise_bands': noise_bands,
            'f0_hz': controls['f0_hz']
            }

        return synth_out

class Reverb(nn.Module):
    def __init__(self, length, sample_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sample_rate = sample_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sample_rate
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


if __name__ == "__main__":
    fmsynth_string = FMSynth(is_reverb=False, sample_rate=16000, block_size=64)
    freq = 523.25
    controls = {}
    controls['f0_hz'] = torch.ones(1, 1000, 1) * freq
    controls['ol'] = torch.zeros(1, 1000, 6)

    synth_out = fmsynth_string(controls)['synth_audio']

    signal = synth_out.squeeze().cpu().detach().numpy()
    # signal_gt, sr = librosa.load("dexed_output_ol50_coarse1.wav", sr=16000)
    # signal_gt = signal_gt / np.amax(signal_gt)
    # # print(signal_gt.shape)

    # plt.plot(signal[16000:16400], label="signal")
    # plt.plot(signal_gt[16001:16401], label="signal_gt")
    # plt.legend()
    # plt.show()

    # print(np.amax(signal_gt))

    # S_dexed = np.abs(librosa.stft(signal_gt))
    # S_test = np.abs(librosa.stft(signal))

    # fig, ax = plt.subplots()
    # import librosa.display
    # img = librosa.display.specshow(librosa.amplitude_to_db(S_dexed,
    #                                                     ref=np.max),
    #                             y_axis='log', x_axis='time', ax=ax)
    # ax.set_title('Dexed')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    # plt.show()

    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(librosa.amplitude_to_db(S_test,
    #                                                     ref=np.max),
    #                             y_axis='log', x_axis='time', ax=ax)
    # ax.set_title('Test')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    # plt.show()

    sf.write("dexed_test_92.wav", signal, 16000)

