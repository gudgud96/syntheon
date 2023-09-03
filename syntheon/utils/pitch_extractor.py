"""
[WIP] Common class for pitch extraction across all synthesizers.
"""

import numpy as np
import os
import crepe
from torchcrepeV2 import TorchCrepePredictor
import yaml 

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../inferencer/vital/config.yaml"
    ), 'r'
) as stream:
    config = yaml.safe_load(stream)

device = config["device"]
if device == "cuda":
    crepe_predictor = TorchCrepePredictor()
else:
    crepe_predictor = TorchCrepePredictor(device="cpu")


# TODO: use ONNX runtime to enable inference optimization to reduce latency
def extract_pitch(signal, sampling_rate, block_size, model_capacity="full"):
    length = signal.shape[-1] // block_size
    if device == "cpu":
        # use TF crepe for cpu as hardware acceleration
        f0 = crepe.predict(
            signal,
            sampling_rate,
            step_size=int(1000 * block_size / sampling_rate),
            verbose=1,
            center=True,
            viterbi=True,
            model_capacity="full"
        )
        f0 = f0[1].reshape(-1)[:-1]
    else:
        # use torchcrepe for gpu
        f0 = crepe_predictor.predict(
            signal,
            sampling_rate
        )

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0