from syntheon.inferencer.inferencer import Inferencer, InferenceInput, InferenceOutput
from syntheon.inferencer.dexed.models.preprocessor import ProcessData, F0LoudnessRMSPreprocessor
from syntheon.inferencer.dexed.models.ddx7.models import DDSP_Decoder, TCNFMDecoder
from syntheon.inferencer.dexed.models.ddx7.synth import FMSynth
from syntheon.inferencer.dexed.models.amp_utils import *
from syntheon.converter.dexed.dexed_converter import DexedConverter
from syntheon.utils.pitch_extractor import extract_pitch
import yaml
import torch
import librosa
import soundfile as sf
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


class DexedInferenceOutput(InferenceOutput):
    def __init__(self):
        InferenceOutput.__init__(self)
        self.synth_audio = None           # TODO: can put default values here
        self.ol = None


class DexedInferenceInput(InferenceInput):
    def __init__(self):
        self.x = None


class DexedInferencer(Inferencer):
    def convert(self, audio_fname, model_pt_fname=None, enable_eval=False):
        # TODO: convert should be more like framework. preprocess -> load_model -> inference -> post_process
        if model_pt_fname is None:
            model_pt_fname = "syntheon/inferencer/dexed/checkpoints/state_best.pth"
        with open("syntheon/inferencer/dexed/models/conf/data_config.yaml", 'r') as f:
            data_config = yaml.safe_load(f)
        
        preprocessor = ProcessData(
            silence_thresh_dB=data_config["data_processor"]["silence_thresh_dB"], 
            sr=data_config["data_processor"]["sr"], 
            device=data_config["data_processor"]["device"], 
            seq_len=data_config["data_processor"]["seq_len"],
            crepe_params=data_config["data_processor"]["crepe_params"], 
            loudness_params=data_config["data_processor"]["loudness_params"],
            rms_params=data_config["data_processor"]["rms_params"], 
            hop_size=data_config["data_processor"]["hop_size"], 
            max_len=data_config["data_processor"]["max_len"], 
            center=data_config["data_processor"]["center"]
        )

        audio, _ = librosa.load(audio_fname, sr=data_config["data_processor"]["sr"])

        f0 = extract_pitch(audio, data_config["data_processor"]["sr"], block_size=64)
        f0 = f0.astype(np.float32)
        loudness = preprocessor.calc_loudness(audio)
        rms = preprocessor.calc_rms(audio)

        scaler = F0LoudnessRMSPreprocessor()
        x = {
            "audio": torch.tensor(audio).unsqueeze(0).unsqueeze(-1),
            "f0": torch.tensor(f0).unsqueeze(0).unsqueeze(-1),
            "loudness": torch.tensor(loudness).unsqueeze(0).unsqueeze(-1),
            "rms": torch.tensor(rms).unsqueeze(0).unsqueeze(-1)
        }
        scaler.run(x)

        inference_input = DexedInferenceInput()
        inference_input.x = x

        model = self.load_model(model_pt_fname, self.device)
        inference_output = self.inference(model, inference_input, self.device, enable_eval=enable_eval)
        synth_params_dict = self.convert_to_preset(inference_output)
        return synth_params_dict, inference_output.eval_dict

    def load_model(self, model_pt_fname, device="cuda"):
        with open("syntheon/inferencer/dexed/models/conf/recipes/model/tcnres_f0ld_fmstr_noreverb.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # prepare model
        decoder = TCNFMDecoder(n_blocks=config["decoder"]["n_blocks"], 
                                hidden_channels=config["decoder"]["hidden_channels"], 
                                out_channels=config["decoder"]["out_channels"],
                                kernel_size=config["decoder"]["kernel_size"],
                                dilation_base=config["decoder"]["dilation_base"],
                                apply_padding=config["decoder"]["apply_padding"],
                                deploy_residual=config["decoder"]["deploy_residual"],
                                input_keys=config["decoder"]["input_keys"])

        synth = FMSynth(sample_rate=config["synth"]["sample_rate"],
                        block_size=config["synth"]["block_size"],
                        fr=config["synth"]["fr"],
                        max_ol=config["synth"]["max_ol"],
                        synth_module=config["synth"]["synth_module"],
                        is_reverb=False)

        model = DDSP_Decoder(decoder, synth)
        if device == "cuda":
            model.load_state_dict(torch.load(model_pt_fname))
            model.cuda()
        else:
            model.load_state_dict(torch.load(model_pt_fname, map_location=torch.device('cpu')))
        model.eval()        
        return model
    
    def inference(self, model, inference_input, device="cuda", enable_eval=False):
        if device == "cuda":
            inference_input.audio = inference_input.x["audio"].cuda()
            inference_input.f0 = inference_input.x["f0"].cuda()
            inference_input.loudness = inference_input.x["loudness"].cuda()
            inference_input.rms = inference_input.x["rm"].cuda()
        
        # forward pass
        synth_out = model(inference_input.x)

        inference_output = DexedInferenceOutput()
        inference_output.synth_audio = synth_out["synth_audio"]
        inference_output.ol = synth_out["ol"]

        return inference_output
    
    def convert_to_preset(self, inference_output):

        dx_converter = DexedConverter()
        params_dict = dx_converter.serializeToDict("syntheon/inferencer/dexed/Dexed_01.syx")

        lst = []
        for idx in range(6):
            ol = inference_output.ol[0, :, idx]
            ol = ol.cpu().detach().numpy()
            ol = ol.reshape(-1, 5).mean(axis=1)

            # TODO: these are all hacky code...
            if (idx == 0 or idx == 2):
                ol = ol / 0.32
            
            lst.append(np.mean(ol))

        lst = [amplitude_to_dexed_ol(k) for k in lst]

        params_dict[0]["5_OL"] = lst[0]
        params_dict[0]["4_OL"] = lst[1]
        params_dict[0]["3_OL"] = lst[2]
        params_dict[0]["2_OL"] = lst[3]
        params_dict[0]["1_OL"] = lst[4]
        params_dict[0]["0_OL"] = lst[5]
        params_dict[0]["NAME CHAR 1"] = 83
        params_dict[0]["NAME CHAR 2"] = 89
        params_dict[0]["NAME CHAR 3"] = 78
        params_dict[0]["NAME CHAR 4"] = 84
        params_dict[0]["NAME CHAR 5"] = 72
        params_dict[0]["NAME CHAR 6"] = 69
        params_dict[0]["NAME CHAR 7"] = 79
        params_dict[0]["NAME CHAR 8"] = 78
        params_dict[0]["NAME CHAR 9"] = 32
        params_dict[0]["NAME CHAR 10"] = 32

        return params_dict


if __name__ == "__main__":
    # TODO: move to test folder
    dexed_inferencer = DexedInferencer(device="cpu")
    params = dexed_inferencer.convert("test/test_audio/dexed_test_audio_1.wav")

    from syntheon.converter.dexed.dexed_converter import DexedConverter
    dexed_converter = DexedConverter()
    dexed_converter.dict = params
    dexed_converter.parseToPluginFile("dexed_output.syx")