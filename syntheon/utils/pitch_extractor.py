"""
[WIP] Common class for pitch extraction across all synthesizers.
"""

import numpy as np
import os
from torchcrepeV2 import ONNXTorchCrepePredictor
import yaml 

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../inferencer/vital/config.yaml"
    ), 'r'
) as stream:
    config = yaml.safe_load(stream)


crepe_predictor = ONNXTorchCrepePredictor()


def extract_pitch(signal, sampling_rate, block_size, model_capacity="full"):
    length = signal.shape[-1] // block_size
    f0 = crepe_predictor.predict(
        audio=signal, 
        sr=sampling_rate, 
        viterbi=True, 
        center=True, 
        step_size=int(1000 * block_size / sampling_rate),
    )

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0