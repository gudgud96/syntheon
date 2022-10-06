"""
ADSR envelope shaper
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import yaml


with open("syntheon/inferencer/vital/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
device = config["device"]


def soft_clamp_min(x, min_v, T=100):
    return torch.sigmoid((min_v-x)*T)*(min_v-x)+x


class DiffRoundFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input * 10 ** 2) / (10 ** 2)     # because 2 decimal point, 0.01 is the minimum ratio

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ADSREnvelopeShaper(nn.Module):
    def __init__(self, is_round_secs=False):
        super(ADSREnvelopeShaper, self).__init__()
        self.attack_secs = torch.tensor([0])
        self.attack_power = torch.tensor([0])
        self.decay_secs = torch.tensor([0])
        self.decay_power = torch.tensor([0])
        self.sustain_level = torch.tensor([0])
        self.release_secs = torch.tensor([0])
        self.release_power = torch.tensor([0])

        self.is_round_secs = is_round_secs
        self.round_decimal_places = 2   # because block size = 100, so min resolution = 1/ 100
    
    def power_function(self, x, pow=2):
        if pow > 0: # convex
            # transpose
            plt.plot(x.squeeze().detach().numpy(), label='test1')

            if x.squeeze()[0] > x.squeeze()[-1]:
                y_intercept = x.squeeze()[-1]
                y = x - x[:, -1, :]
                max_val = y.squeeze()[0]
                y = y / max_val
            else:
                y_intercept = x.squeeze()[0]
                y = x - x[:, 0, :]
                max_val = y.squeeze()[-1]
                y = y / max_val
            
            y = y ** pow

            # transpose back
            y = y * max_val + y_intercept

            plt.plot(y.squeeze().detach().numpy(), label='test3')
            plt.show()

        else:
            # transpose
            plt.plot(x.squeeze().detach().numpy(), label='test1')

            if x.squeeze()[0] > x.squeeze()[-1]:
                max_val = x.squeeze()[0]
                y = x - x[:, 0, :]
                y_intercept = y.squeeze()[-1]
                y = y / -y_intercept
            else:
                max_val = x.squeeze()[-1]
                y = x - x[:, -1, :]
                y_intercept = y.squeeze()[0]
                y = y / -y_intercept

            plt.plot(y.squeeze().detach().numpy(), label='test2')

            y = -(y ** -pow)

            plt.plot(y.squeeze().detach().numpy(), label='test3')

            # transpose back
            y = y * -y_intercept + max_val

            plt.plot(y.squeeze().detach().numpy(), label='test4')
            plt.legend()
            plt.show()

            # plt.plot(y.squeeze().detach().numpy(), label='test3')
            # plt.show()

        return y
    
    def gen_envelope(self, attack, decay, sus_level, release,
                     floor=None, peak=None, n_frames=250, pow=2):
        """generate envelopes from parameters
        Args:
            floor (torch.Tensor): floor level of the signal 0~1, 0=min_value (batch, 1, channels)
            peak (torch.Tensor): peak level of the signal 0~1, 1=max_value (batch, 1, channels)
            attack (torch.Tensor): relative attack point 0~1 (batch, 1, channels)
            decay (torch.Tensor): actual decay point is attack+decay (batch, 1, channels)
            sus_level (torch.Tensor): sustain level 0~1 (batch, 1, channels)
            release (torch.Tensor): release point is attack+decay+release (batch, 1, channels)
            note_off (float or torch.Tensor, optional): note off position. Defaults to 0.8.
            n_frames (int, optional): number of frames. Defaults to None.
        Returns:
            torch.Tensor: envelope signal (batch_size, n_frames, 1)
        """
        if floor is None:
            floor = torch.tensor([0.]).unsqueeze(0).unsqueeze(-1)
            if device == "cuda":
                floor = floor.cuda()
        if peak is None:
            peak = torch.tensor([1.]).unsqueeze(0).unsqueeze(-1)
            if device == "cuda":
                peak = peak.cuda()
        
        # attack ratio can be 0, but the initial adsr 0 values should be epsilon-ed
        # decay ratio should be larger than 1 / total_n_frames
        # decay secs should be larger than 1 / block_size
        # attack and decay secs will be rounded up to minimum resolution (min resolution = 1/ block_size)
        # DO WE REALLY NEED THIS? 
        MIN_RATIO = 1 / 400     # HACK, 4 seconds * block size 100
        
        attack = torch.clamp(attack, min=0, max=1)
        decay = torch.clamp(decay, min=0, max=1)
        sus_level = torch.clamp(sus_level, min=0.001, max=1)
        release = torch.clamp(release, min=0, max=1)

        batch_size = attack.shape[0]
        if n_frames is None:
            n_frames = self.n_frames
        # batch, n_frames, 1
        x = torch.linspace(0, 1.0, n_frames)[None, :, None].repeat(batch_size, 1, 1)
        x[:, 0, :] = 1e-6       # offset 0 to epsilon value, so when attack = 0, first adsr value is not 0 but 1
        x = x.to(attack.device)

        A = x / (attack + 1e-6)
        # A = self.power_function(A, pow=2)
        A = torch.clamp(A, max=1.0)

        D = (x - attack) * (sus_level - 1) / (decay+1e-6)
        # D = self.power_function(D, pow=-2.7)
        D = torch.clamp(D, max=0.0)
        D = soft_clamp_min(D, sus_level-1)

        S = (x - 1) * (-sus_level / (release+1e-6))
        S = torch.clamp(S, max=0.0)
        S = soft_clamp_min(S, -sus_level)

        signal = (A + D + S) * (peak - floor) + floor
        return torch.clamp(signal, min=0., max=1.)
    
    def forward(self, 
                attack_secs,
                decay_secs,
                sustain_level,
                block_size=100, 
                sr=44100, 
                total_secs=8):
        if self.is_round_secs:
            attack_secs = DiffRoundFunc.apply(attack_secs)
            decay_secs = DiffRoundFunc.apply(decay_secs)

        self.attack_secs = attack_secs
        self.decay_secs = decay_secs
        self.sustain_level = sustain_level
        
        attack_ratio = attack_secs / total_secs
        decay_ratio = decay_secs / total_secs
        # TODO: parameteize release_ratio
        release_ratio = torch.tensor([0.]).repeat(attack_secs.size(0), 1, 1)
        if device == "cuda":
            release_ratio = release_ratio.cuda()

        attack_ratio = attack_ratio.unsqueeze(-1).unsqueeze(-1)
        decay_ratio = decay_ratio.unsqueeze(-1).unsqueeze(-1)
        sus_level = sustain_level.unsqueeze(-1).unsqueeze(-1)

        signal = self.gen_envelope(attack_ratio, decay_ratio, sus_level, release_ratio,
                                    floor=None, peak=None, n_frames=int(total_secs * block_size),
                                    pow=2)
        return signal.squeeze()


def get_amp_shaper(
                shaper, 
                onsets, 
                attack_secs,
                decay_secs,
                sustain_level,
                offsets=None):
    """
    implement case with no offset first.
    """
    if offsets is None:
        # if offset not specified, take next onset as offset
        offsets = onsets[1:]
        onsets = onsets[:len(onsets) - 1]

    start_offset = int(onsets[0] * 100)        # TODO: 100 is block size
    onsets, offsets = torch.tensor(onsets), torch.tensor(offsets)
    if device == "cuda":
        onsets, offsets = onsets.cuda(), offsets.cuda()
    dur_vec = offsets - onsets
    lst = []

    # append zeros first before first onset
    if device == "cuda":
        lst.append(torch.zeros(start_offset).cuda())
    else:
        lst.append(torch.zeros(start_offset))

    for dur in dur_vec:
        dur = round(dur.item(), 2)
        adsr = shaper(
            attack_secs=torch.tensor([0.2, 0.1]), 
            decay_secs=torch.tensor([0.1, 0.2]),
            sustain_level=torch.tensor([0.9, 0.2]),
            total_secs=dur)
        lst.append(adsr[0].squeeze())   # TODO: fix the batch size case
    
    final_signal = torch.cat(lst, dim=0)
    return final_signal


def get_amp_shaper_v2(
                shaper, 
                onsets, 
                attack_secs,
                decay_secs,
                sustain_level,
                offsets=None):
    """
    implement case with no offset first. enable batches
    """
    if offsets is None:
        # if offset not specified, take next onset as offset
        offsets = onsets[1:]
        onsets = onsets[:len(onsets) - 1]

    start_offset = int(onsets[0] * 100)        # TODO: 100 is block size
    onsets, offsets = torch.tensor(onsets), torch.tensor(offsets)
    if device == "cuda":
        onsets, offsets = onsets.cuda(), offsets.cuda()
    dur_vec = offsets - onsets
    lst = []

    # append zeros first before first onset
    if device == "cuda":
        lst.append(torch.zeros(start_offset).cuda())
    else:
        lst.append(torch.zeros(start_offset))

    for dur in dur_vec:
        dur = round(dur.item(), 2)
        adsr = shaper(
            attack_secs=attack_secs, 
            decay_secs=decay_secs,
            sustain_level=sustain_level,
            total_secs=dur)
        
        # adsr shape should be (bs, dur * block_size)
        lst.append(adsr)
    
    final_signal = torch.cat(lst, dim=-1)
    return final_signal


if __name__ == "__main__":
    # TODO: unit test for this class
    shaper = ADSREnvelopeShaper(is_round_secs=False)
    adsrs = []
    for elem in [0.0, 0.001, 0.005, 0.01, 0.02]:
        attack_secs, decay_secs, sustain_level = torch.tensor([0.2]), torch.tensor([elem]), torch.tensor([0.8])
        if device == "cuda": 
            attack_secs = attack_secs.cuda()
            decay_secs = decay_secs.cuda()
            sustain_level = sustain_level.cuda()
        
        x2 = shaper(
            attack_secs=attack_secs, 
            decay_secs=decay_secs,
            sustain_level=sustain_level,
            total_secs=4)

        adsrs.append(x2.squeeze().cpu().detach().numpy()[:30])

    for idx, elem in enumerate([0.0, 0.001, 0.005, 0.01, 0.02]):
        plt.plot(adsrs[idx], label=str(elem))
        plt.scatter(range(30), adsrs[idx])
    plt.legend()
    plt.show()