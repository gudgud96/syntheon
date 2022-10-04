"""
Common class for pitch extraction.
"""

import numpy as np
import crepe
from torchcrepeV2 import TorchCrepePredictor
import yaml 

with open("syntheon/inferencer/vital/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

device = config["device"]
if device == "cuda":
    crepe_predictor = TorchCrepePredictor()
else:
    crepe_predictor = TorchCrepePredictor(device="cpu")


def extract_pitch(signal, sampling_rate, block_size, model_capacity="full"):
    length = signal.shape[-1] // block_size
    if device == "cpu":
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