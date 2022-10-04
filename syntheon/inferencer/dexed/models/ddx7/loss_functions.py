import ddx7.core as core
import torch
import torch.nn as nn
import torchcrepe
from functools import partial


'''
Asimetric L1 distance
'''
def asim_l1_distance(a,b,alpha=1,beta=1):
    diff = a-b
    pos_diff = diff * (diff > 0)
    neg_diff = diff * (diff < 0)
    as_diff = alpha * pos_diff + beta * neg_diff
    as_mse = torch.abs(as_diff).mean()
    return as_mse


def asim_msfft_loss(a1,
                    a2,
                    scales=[4096, 2048, 1024, 512, 256, 128],
                    overlap=0.75,
                    alpha=1,
                    beta=1):
    '''
    DDSP Original MS FFT loss with lin + log spectra analysis
    '''
    if(len(a1.size()) == 3):
        a1 = a1.squeeze(-1)
    if(len(a2.size()) == 3):
        a2 = a2.squeeze(-1)
    ori_stft = core.multiscale_fft(
        a1,
        scales,
        overlap,
    )
    rec_stft = core.multiscale_fft(
        a2,
        scales,
        overlap,
    )

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = asim_l1_distance(s_x, s_y,alpha,beta)
        log_loss = asim_l1_distance(core.safe_log(s_x),core.safe_log(s_y),alpha,beta)
        loss = loss + lin_loss + log_loss

    return loss



def ddsp_msfft_loss(a1,
                    a2,
                    scales=[4096, 2048, 1024, 512, 256, 128],
                    overlap=0.75):
    '''
    DDSP Original MS FFT loss with lin + log spectra analysis
        Some remarks: the stfts have to be normalized otherwise the netowrk weights different excerpts to different importance.
                      We compute the mean of the L1 difference between normalized magnitude spectrograms
                      so that the magnitude of the loss do not change with the window size.
    '''
    if(len(a1.size()) == 3):
        a1 = a1.squeeze(-1)
    if(len(a2.size()) == 3):
        a2 = a2.squeeze(-1)
    ori_stft = core.multiscale_fft(
        a1,
        scales,
        overlap,
    )
    rec_stft = core.multiscale_fft(
        a2,
        scales,
        overlap,
    )

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (core.safe_log(s_x) - core.safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss

    return loss

class rec_loss(nn.Module):
    def __init__(self,scales,overlap,alpha=None,beta=None):
        super().__init__()
        self.scales = scales
        self.overlap = overlap
        if(alpha is not None and beta is not None):
            self.loss_fn = partial(asim_msfft_loss,alpha=alpha,beta=beta)
            print(f'[INFO] rec_loss() - Using asimetrical reconstruction loss. alpha: {alpha} - beta: {beta}')
        else:
            self.loss_fn = ddsp_msfft_loss
    def forward(self,ref,synth):
        return self.loss_fn(ref,synth,
                    self.scales,
                    self.overlap)
