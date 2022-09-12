"""
Connects model output to synth preset parameter IR.
"""

class Inferencer:
    def __init__(self, device="cuda"):
        self.device = device

    def convert(self, model_pt_fname, audio_fname, output_name):
        attention, wt_output = self.load_model(model_pt_fname, audio_fname)
        self.convert_to_preset(attention, wt_output, output_name)
    
    def load_model(self, model_pt_fname, audio_fname):
        attention, wt_output = inference(model_pt_fname, audio_fname, device=self.device)
        return attention, wt_output

    def convert_to_preset(self, attention, wt_output, output_name):
        return NotImplementedError