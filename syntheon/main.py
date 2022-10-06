"""
Function APIs to be called externally.
"""
from .converter.dexed.dexed_converter import DexedConverter
from .inferencer.dexed.dexed_inferencer import DexedInferencer
from .converter.vital.vital_converter import VitalConverter
from .inferencer.vital.vital_inferencer import VitalInferencer


obj_dict = {
    "dexed": {
        "converter": DexedConverter,
        "inferencer": DexedInferencer
    },
    "vital": {
        "converter": VitalConverter,
        "inferencer": VitalInferencer
    }
}

def infer_params(input_audio_name, synth_name, enable_eval=False):
    if synth_name not in obj_dict:
        raise ValueError("Synth name {} not available for parameter inference".format(synth_name))
    
    inferencer = obj_dict[synth_name]["inferencer"](device="cpu")
    params, eval_dict = inferencer.convert(input_audio_name, enable_eval=enable_eval)

    converter = obj_dict[synth_name]["converter"]()
    converter.dict = params
    output_fname = "{}_output.syx".format(synth_name)
    converter.parseToPluginFile(output_fname)

    return output_fname, eval_dict