from syntheon.inferencer.inferencer import Inferencer, InferenceInput, InferenceOutput
from syntheon.inferencer.vital.models.model import WTS
from syntheon.inferencer.vital.models.preprocessor import *
from syntheon.converter.vital.vital_constants import N_WAVETABLES, CUSTOM_KEYS
import yaml 
import torch
import numpy as np
import json

with open("syntheon/inferencer/vital/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
sr = config["common"]["sampling_rate"]
block_size = config["common"]["block_size"]
duration_secs = config["common"]["duration_secs"]
batch_size = config["train"]["batch_size"]
scales = config["train"]["scales"]
overlap = config["train"]["overlap"]
hidden_size = config["train"]["hidden_size"]
n_harmonic = config["train"]["n_harmonic"]
n_bands = config["train"]["n_bands"]
n_wavetables = config["train"]["n_wavetables"]
n_mfcc = config["train"]["n_mfcc"]
train_lr = config["train"]["start_lr"]
visualize = config["visualize"]
device = config["device"]
signal_length = sr * 3


class VitalInferenceOutput(InferenceOutput):
    def __init__(self):
        self.wt_output = None           # TODO: can put default values here
        self.attention_output = None


class VitalInferenceInput(InferenceInput):
    def __init__(self):
        self.y = None
        self.pitch = None
        self.loudness = None
        self.times = None
        self.onset_frames = None
        self.mfcc = None


class VitalInferencer(Inferencer):
    def convert(self, audio_fname, model_pt_fname=None):
        # for vital, the model loading depends on signal input length
        # TODO: should not let this happen ,signal input length become a config
        if model_pt_fname is None:
            model_pt_fname = "syntheon/inferencer/vital/checkpoints/model_ableton_3.pt"
        y, y_len, pitch, loudness, times, onset_frames, mfcc = preprocess(audio_fname, sampling_rate=16000, block_size=160, 
                                                                         signal_length=signal_length)
        inference_input = VitalInferenceInput()
        inference_input.y = y
        inference_input.pitch = pitch
        inference_input.loudness = loudness
        inference_input.times = times
        inference_input.onset_frames = onset_frames
        inference_input.mfcc = mfcc

        model = self.load_model(model_pt_fname, y_len, self.device)
        inference_output = self.inference(model, inference_input, self.device)
        synth_params_dict = self.convert_to_preset(inference_output)
        return synth_params_dict

    def load_model(self, model_pt_fname, length, device="cuda"):
        model = WTS(hidden_size=hidden_size, n_harmonic=n_harmonic, n_bands=n_bands, sampling_rate=sr,
                    block_size=block_size, n_wavetables=3, mode="wavetable", 
                    duration_secs=length // sr,
                    adsr_hidden_size=signal_length // 40)
        if device == "cuda":
            model.load_state_dict(torch.load(model_pt_fname))
            model.cuda()
        else:
            model.load_state_dict(torch.load(model_pt_fname, map_location=torch.device('cpu')))
        model.eval()        
        return model
    
    def inference(self, model, inference_input, device="cuda"):
        if device == "cuda":
            inference_input.y = inference_input.y.cuda()
            inference_input.mfcc = inference_input.mfcc.cuda()
            inference_input.pitch = inference_input.pitch.cuda()
            inference_input.loudness = inference_input.loudness.cuda()

        # forward pass
        _, _, _, attention_output = model(
            inference_input.y, 
            inference_input.mfcc, 
            inference_input.pitch, 
            inference_input.loudness, 
            inference_input.times, 
            inference_input.onset_frames
        )

        # write wavetables to numpy file
        wt_output = []

        # interp from 512 to 2048
        output_length = 2048
        for i in range(N_WAVETABLES):
            wt = model.wts.wavetables[i].cpu().detach().numpy().squeeze()
            wt_interp = np.interp(
                np.linspace(0, 1, output_length, endpoint=False),
                np.linspace(0, 1, wt.shape[0], endpoint=False),
                wt,
            )
            wt_output.append(wt_interp)

        wt_output = np.stack(wt_output, axis=0)
        attention_output = attention_output.cpu().detach().numpy().squeeze()

        inference_output = VitalInferenceOutput()
        inference_output.wt_output = wt_output
        inference_output.attention_output = attention_output

        return inference_output
    
    def convert_to_preset(self, inference_output):
        with open("syntheon/inferencer/vital/init.vital") as f:
            x = json.load(f)

        x[CUSTOM_KEYS] = {}
        x[CUSTOM_KEYS]["wavetables"] = []
        for idx in range(N_WAVETABLES):
            cur_dict = {
                "name": "Litmus WT {}".format(idx),
                "wavetable": inference_output.wt_output[idx],
                "osc_level": inference_output.attention_output[idx].item()
            }
            x[CUSTOM_KEYS]["wavetables"].append(cur_dict)

        return x


if __name__ == "__main__":
    # TODO: move to test folder
    vital_inferencer = VitalInferencer(device="cpu")
    params = vital_inferencer.convert("syntheon/inferencer/vital/checkpoints/model_ableton_3.pt", "test/test_audio/vital_test_audio_1.wav")

    from syntheon.converter.vital.vital_converter import VitalConverter
    vital_converter = VitalConverter()
    vital_converter.dict = params
    vital_converter.parseToPluginFile("vital_output.vital")

    