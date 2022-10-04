import torch
import torch.nn as nn
from syntheon.inferencer.dexed.models.ddx7.core import get_gru, get_mlp
import torch.nn.functional as F

'''
Wrapper class for either HpN or DDX7
'''
class DDSP_Decoder(nn.Module):
    def __init__(self,decoder,synth):
        super().__init__()
        net = []
        net.append(decoder)
        net.append(synth)
        self.net = nn.Sequential(*net)

    def forward(self,x):
        return self.net(x)

    def get_sr(self):
        return self.net[1].sample_rate

    def enable_cumsum_nd(self):
        self.net[1].use_cumsum_nd=True

    def get_params(self,param):
        if(param == 'reverb_decay'):
            return self.net[1].reverb.decay.item()
        if(param == 'reverb_wet'):
            return self.net[1].reverb.wet.item()

'''
GRU-Based decoder for HpN Baseline
'''
class RnnFCDecoder(nn.Module):
    def __init__(self, hidden_size=512, sample_rate=16000,
                 input_keys=None,input_sizes=[1,1,16],
                 output_keys=['amplitude','harmonic_distribution','noise_bands'],
                 output_sizes=[1,100,65]):
        super().__init__()
        self.input_keys = input_keys
        self.input_sizes = input_sizes
        n_keys = len(input_keys)
        # Generate MLPs of size: in_size: 1 ; n_layers = 3 (with layer normalization and leaky relu)
        if(n_keys == 2):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], hidden_size, 3),
                                          get_mlp(input_sizes[1], hidden_size, 3)])
        elif(n_keys == 3):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], hidden_size, 3),
                                          get_mlp(input_sizes[1], hidden_size, 3),
                                          get_mlp(input_sizes[2], hidden_size, 3)])
        else:
            raise ValueError("Expected 2 or 3 input keys. got: {}".format(input_keys))

        #Generate GRU: input_size = n_keys * hidden_size ; n_layers = 1 (that's the default config)
        self.gru = get_gru(n_keys, hidden_size)

        #Generate output MLP: in_size: hidden_size + 2 ; n_layers = 3
        self.out_mlp = get_mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = []
        self.output_keys = output_keys
        self.output_sizes = output_sizes
        for v,k in enumerate(output_keys):
            self.proj_matrices.append(nn.Linear(hidden_size,output_sizes[v]))

        self.proj_matrices = nn.ModuleList(self.proj_matrices)
        self.sample_rate = sample_rate

    def forward(self, x):
        # Run pitch and loudness and z (if available) inputs through the respectives input MLPs.
        # Then, concatenate the outputs in a flat vector.

        # Run through input_keys and load inputs accordingly
        hidden = torch.cat([self.in_mlps[v](x[k]) for v,k in enumerate(self.input_keys)],-1)

        # Run the flattened vector through the GRU.
        # The GRU predicts the embedding.
        # Then, concatenate the embedding with the disentangled parameters of pitch and loudness (nhid+2 size vector)
        hidden = torch.cat([self.gru(hidden)[0], x['f0_scaled'], x['loudness_scaled']], -1)
        # Run the embedding through the output MLP to obtain a 512-sized output vector.
        hidden = self.out_mlp(hidden)


        # Run embedding through a projection_matrix to get outputs
        controls = {}
        for v,k in enumerate(self.output_keys):
            controls[k] = self.proj_matrices[v](hidden)

        controls['f0_hz'] = x['f0']

        return controls

'''
TCN-Based decoder for DDX7
'''
class TCNFMDecoder(nn.Module):
    '''
    FM Decoder with sigmoid output
    '''
    def __init__(self,n_blocks=2,hidden_channels=64,out_channels=6,
                kernel_size=3,dilation_base=2,apply_padding=True,
                deploy_residual=False,
                input_keys=None,z_size=None,
                output_complete_controls=True):
        super().__init__()

        # Store receptive field
        dilation_factor = (dilation_base**n_blocks-1)/(dilation_base-1)
        self.receptive_field = 1 + 2*(kernel_size-1)*dilation_factor
        print("[INFO] TCNFNDecoder - receptive field is: {}".format(self.receptive_field))

        self.input_keys = input_keys
        n_keys = len(input_keys)
        self.output_complete_controls = output_complete_controls

        if(n_keys == 2):
            in_channels = 2
        elif(n_keys == 3):
            in_channels = 2 + z_size
        else:
            raise ValueError("Expected 2 or 3 input keys. got: {}".format(input_keys))

        base = 0
        net = []

        net.append(TCN_block(in_channels,hidden_channels,hidden_channels,kernel_size,
            dilation=dilation_base**base,apply_padding=apply_padding,
            deploy_residual=deploy_residual))
        if(n_blocks>2):
            for i in range(n_blocks-2):
                base += 1
                net.append(TCN_block(hidden_channels,hidden_channels,hidden_channels,
                    kernel_size,dilation=dilation_base**base,apply_padding=apply_padding))

        base += 1
        net.append(TCN_block(hidden_channels,hidden_channels,out_channels,kernel_size,
            dilation=dilation_base**base,apply_padding=apply_padding,
            deploy_residual=deploy_residual,last_block=True))

        self.net = nn.Sequential(*net)

    def forward(self,x):
        # Reshape features to follow Conv1d convention (nb,ch,seq_Len)
        conditioning = torch.cat([x[k] for v,k in enumerate(self.input_keys)],-1).permute([0,-1,-2])

        ol = self.net(conditioning)
        ol = ol.permute([0,-1,-2])
        if self.output_complete_controls is True:
            synth_params = {
                'f0_hz': x['f0'], #In Hz
                'ol': ol
                }
        else:
            synth_params = ol
        return synth_params

class TCN_block(nn.Module):
    '''
    TCN Block
    '''
    def __init__(self,in_channels,hidden_channels,out_channels,
                kernel_size,stride=1,dilation=1,apply_padding=True,
                last_block=False,deploy_residual=False):
        super().__init__()
        block = []
        cnv1 = CausalConv1d(in_channels,hidden_channels,kernel_size,
            stride=stride,dilation=dilation,apply_padding=apply_padding)
        block.append(torch.nn.utils.weight_norm( cnv1 ) )
        block.append(nn.ReLU())
        block.append(nn.Dropout())

        cnv2 = CausalConv1d(hidden_channels,out_channels,kernel_size,
            stride=stride,dilation=dilation,apply_padding=apply_padding)
        block.append(torch.nn.utils.weight_norm( cnv2 ) )
        if(last_block == False):
            block.append(nn.ReLU())
            block.append(nn.Dropout())

        self.block = nn.Sequential(*block)
        self.residual = None
        if(deploy_residual):
            if(apply_padding):
                self.residual = nn.Conv1d(in_channels,out_channels,1,padding = 0,stride=stride)
            else:
                raise ValueError("Residual connection is only possible when padding is enabled.")

    def forward(self,data):
        block_out = self.block(data)
        if(self.residual is not None):
            residual = self.residual(data)
            block_out = block_out + residual
        return block_out


class CausalConv1d(torch.nn.Conv1d):
    '''
    Basic layer for implementing a TCN
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 apply_padding=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.apply_padding = apply_padding
        self.__padding = dilation*(kernel_size - 1)

    def forward(self, input):
        # Apply left padding using torch.nn.functional and then compute conv.
        if(self.apply_padding):
            return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))
        else:
            return super(CausalConv1d, self).forward(input)
