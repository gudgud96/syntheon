"""
Connects model output to synth preset parameter IR.
"""
class InferenceInput:
    def __init__(self):
        return NotImplementedError


class InferenceOutput:
    def __init__(self):
        # for storing evaluation results
        self.eval_dict = {
            "loss":  -1
        }


class Inferencer:
    def __init__(self, device="cuda"):
        self.device = device

    def convert(self, model_pt_fname, audio_fname):
        model = self.load_model(model_pt_fname, self.device)
        inference_output = self.inference(model, audio_fname, self.device)
        synth_params_dict = self.convert_to_preset(inference_output)
        return synth_params_dict, inference_output.eval_dict
    
    def load_model(self, model_pt_fname, device="cuda"):
        return NotImplementedError
        
    def inference(self, model, audio_fname):
        return NotImplementedError

    def convert_to_preset(self, inference_output):
        return NotImplementedError