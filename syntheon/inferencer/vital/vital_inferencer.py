from syntheon.inferencer.inferencer import Inferencer, InferenceInput, InferenceOutput
from syntheon.inferencer.vital.models.model_v2 import WTSv2
from syntheon.inferencer.vital.models.preprocessor import *
from syntheon.inferencer.vital.models.core import multiscale_fft
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
signal_length = sr * 4


class VitalInferenceOutput(InferenceOutput):
    def __init__(self):
        InferenceOutput.__init__(self)
        self.wt_output = None           # TODO: can put default values here
        self.attention_output = None
        self.attack = None
        self.decay = None
        self.sustain = None


class VitalInferenceInput(InferenceInput):
    def __init__(self):
        self.y = None
        self.pitch = None
        self.loudness = None
        self.times = None
        self.onset_frames = None
        self.mfcc = None


class VitalInferencer(Inferencer):
    def convert(self, audio_fname, model_pt_fname=None, enable_eval=False):
        # for vital, the model loading depends on signal input length
        if model_pt_fname is None:
            model_pt_fname = "syntheon/inferencer/vital/checkpoints/model_adsr_loudness_v2.pt"
        
        y, pitch, loudness, times, onset_frames, mfcc = preprocess(audio_fname, sampling_rate=16000, block_size=160, 
                                                                   signal_length=signal_length)
        inference_input = VitalInferenceInput()
        inference_input.y = y
        inference_input.pitch = pitch
        inference_input.loudness = loudness
        inference_input.times = times
        inference_input.onset_frames = onset_frames
        inference_input.mfcc = mfcc

        model = self.load_model(model_pt_fname, self.device)
        inference_output = self.inference(model, inference_input, self.device, enable_eval=enable_eval)
        synth_params_dict = self.convert_to_preset(inference_output)
        return synth_params_dict, inference_output.eval_dict

    def load_model(self, model_pt_fname, device="cuda"):
        model = WTSv2(hidden_size=hidden_size, n_harmonic=n_harmonic, n_bands=n_bands, sampling_rate=sr,
                block_size=block_size,  mode="wavetable", 
                duration_secs=4, num_wavetables=1, wavetable_smoothing=False, preload_wt=True, enable_amplitude=False,
                is_round_secs=False, device=device)
        if device == "cuda":
            model.load_state_dict(torch.load(model_pt_fname))
            model.cuda()
        else:
            model.load_state_dict(torch.load(model_pt_fname, map_location=torch.device('cpu')))
        model.eval()        
        return model
    
    def inference(self, model, inference_input, device="cuda", enable_eval=False):
        if device == "cuda":
            inference_input.y = inference_input.y.cuda()
            inference_input.mfcc = inference_input.mfcc.cuda()
            inference_input.pitch = inference_input.pitch.cuda()
            inference_input.loudness = inference_input.loudness.cuda()

        # forward pass
        with torch.no_grad():
            _, adsr, output, attention_output, wavetables, _, _ = model(
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
            wt = wavetables[i].cpu().detach().numpy().squeeze()
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
        inference_output.attack = adsr[0][0].cpu().detach().numpy().squeeze().item()
        inference_output.decay = adsr[1][0].cpu().detach().numpy().squeeze().item()
        inference_output.sustain = adsr[2][0].cpu().detach().numpy().squeeze().item()

        if enable_eval:
            self.eval(inference_input.y, output, inference_output)

        return inference_output
    
    def convert_to_preset(self, inference_output):
        with open("syntheon/inferencer/vital/init.vital") as f:
            x = json.load(f)

        x[CUSTOM_KEYS] = {}
        x[CUSTOM_KEYS]["wavetables"] = []
        for idx in range(N_WAVETABLES):
            cur_dict = {
                "name": "Litmus WT {}".format(idx + 1),
                "wavetable": inference_output.wt_output[idx],
                "osc_level": inference_output.attention_output[idx].item()
            }
            x[CUSTOM_KEYS]["wavetables"].append(cur_dict)
            x[CUSTOM_KEYS]["adsr"] = {}
            x[CUSTOM_KEYS]["adsr"]["attack"] = inference_output.attack
            x[CUSTOM_KEYS]["adsr"]["attack_power"] = 0.0
            x[CUSTOM_KEYS]["adsr"]["decay"] = inference_output.decay
            x[CUSTOM_KEYS]["adsr"]["decay_power"] = 0.0
            x[CUSTOM_KEYS]["adsr"]["sustain"] = inference_output.sustain

        return x
    
    def eval(self, y, output, inference_output):
        ori_stft = multiscale_fft(
                    y[0].squeeze(),
                    scales,
                    overlap,
                )
        rec_stft = multiscale_fft(
            output[0].squeeze(),
            scales,
            overlap,
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft): 
            lin_loss = ((s_x - s_y).abs()).mean()
            loss += lin_loss
        
        inference_output.eval_dict["loss"] = loss.item()
        inference_output.eval_dict["output"] = output[0].cpu().detach().numpy().squeeze()


if __name__ == "__main__":
    # TODO: move to test folder
    vital_inferencer = VitalInferencer(device="cpu")
    params, eval_dict = vital_inferencer.convert("test/test_audio/vital_test_audio_2.wav", enable_eval=True)

    from syntheon.converter.vital.vital_converter import VitalConverter
    vital_converter = VitalConverter()
    vital_converter.dict = params
    vital_converter.parseToPluginFile("vital_output.vital")

    