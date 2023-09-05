"""
Function APIs to be called externally.
"""
from .converter.vital.vital_converter import VitalConverter
from .inferencer.vital.vital_inferencer import VitalInferencer


obj_dict = {
    "vital": {
        "converter": VitalConverter,
        "inferencer": VitalInferencer,
        "file_ext": "vital"
    }
}

def infer_params(input_audio_name, synth_name, enable_eval=False):
    if synth_name not in obj_dict:
        raise ValueError("Synth name {} not available for parameter inference".format(synth_name))
    
    inferencer = obj_dict[synth_name]["inferencer"](device="cpu")
    params, eval_dict = inferencer.convert(input_audio_name, enable_eval=enable_eval)

    converter = obj_dict[synth_name]["converter"]()
    converter.dict = params
    output_fname = "{}_output.{}".format(synth_name, obj_dict[synth_name]["file_ext"])
    converter.parseToPluginFile(output_fname)

    return output_fname, eval_dict