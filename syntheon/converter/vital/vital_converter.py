import json
import base64
import struct
from syntheon.converter.converter import SynthConverter
from syntheon.converter.vital.vital_constants import N_WAVETABLES, CUSTOM_KEYS
import numpy as np
import math


class Base64Converter:
    def __init__(self):
        pass

    def encode(self, signal):
        signal_bytes = struct.pack('{}f'.format(len(signal)), *signal)
        base64_string = base64.b64encode(signal_bytes)

        return base64_string.decode('ascii')
    
    def decode(self, base64_string, output_length=2048):
        signal_bytes = base64.decodebytes(base64_string.encode('ascii'))
        arr = [k for k in struct.iter_unpack('f', signal_bytes)]    # unpack as 2 bytes integer
        arr = [k[0] for k in arr]                                   # normalize

        return np.array(arr)


class VitalConverter(SynthConverter):
    def __init__(self):
        SynthConverter.__init__(self)
        self.keys = []
        self.base64_converter = Base64Converter()
    
    def serializeToDict(self, fname):
        try:
            with open(fname) as f:
                self.dict = json.load(f)
            
            # decode custom part
            self.dict[CUSTOM_KEYS] = {}
            self.dict[CUSTOM_KEYS]["wavetables"] = []
            for idx in range(len(N_WAVETABLES)):
                wavetable_str = self.dict["settings"]["wavetables"][idx]["groups"][0]["components"][0]["keyframes"][0]["wave_data"]
                wavetable_name = self.dict["settings"]["wavetables"][idx]["name"]
                wavetable_osc_level = self.dict["settings"]["osc_{}_level".format(idx + 1)]
                wavetable = self.base64_converter.decode(wavetable_str)     # return np.array
                cur_dict = {
                    "name": wavetable_name,
                    "wavetable": wavetable,
                    "osc_level": wavetable_osc_level
                }
                self.dict[CUSTOM_KEYS]["wavetables"].append(cur_dict)
            
            # switch off unused wavetables
            if N_WAVETABLES == 1:
                self.dict["settings"]["osc_2_on"] = 0.0
                self.dict["settings"]["osc_3_on"] = 0.0
            elif N_WAVETABLES == 2:
                self.dict["settings"]["osc_3_on"] = 0.0
                    
        except Exception as e:
            print(str(e))
        
        return self.dict
    
    def parseToPluginFile(self, fname):
        """
        vital parameters value scale: https://github.com/mtytel/vital/blob/c0694a193777fc97853a598f86378bea625a6d81/src/common/synth_parameters.cpp
        value scale computation: https://github.com/mtytel/vital/blob/c0694a193777fc97853a598f86378bea625a6d81/src/plugin/value_bridge.h
        """
        # encode custom part
        wavetables = self.dict[CUSTOM_KEYS]["wavetables"]
        for idx in range(N_WAVETABLES):
            wavetable = wavetables[idx]["wavetable"]
            wavetable_name = wavetables[idx]["name"]
            wavetable_osc_level = wavetables[idx]["osc_level"]

            wavetable_str = self.base64_converter.encode(wavetable)
            self.dict["settings"]["wavetables"][idx]["groups"][0]["components"][0]["keyframes"][0]["wave_data"] = wavetable_str
            self.dict["settings"]["wavetables"][idx]["name"] = wavetable_name
            self.dict["settings"]["osc_{}_level".format(idx + 1)] = wavetable_osc_level
        
        # switch off unused wavetables
        if N_WAVETABLES == 1:
            self.dict["settings"]["osc_2_on"] = 0.0
            self.dict["settings"]["osc_3_on"] = 0.0
        elif N_WAVETABLES == 2:
            self.dict["settings"]["osc_3_on"] = 0.0
        
        # adsr filter
        adsrs = self.dict[CUSTOM_KEYS]["adsr"]
        # attack is kQuartic
        self.dict["settings"]["env_1_attack"] = math.sqrt(math.sqrt(adsrs["attack"]))
        # attack power is kLinear
        self.dict["settings"]["env_1_attack_power"] = adsrs["attack_power"]
        # decay is kQuartic
        self.dict["settings"]["env_1_decay"] = math.sqrt(math.sqrt(adsrs["decay"]))
        # decay power is kLinear
        self.dict["settings"]["env_1_decay_power"] = adsrs["decay_power"]
        # sustain is kLinear
        self.dict["settings"]["env_1_sustain"] = adsrs["sustain"]

        # self.dict["settings"]["env_1_delay"] = adsrs["delay"]
        # self.dict["settings"]["env_1_hold"] = adsrs["hold"]
        # self.dict["settings"]["env_1_release"] = adsrs["release"]
        # self.dict["settings"]["env_1_release_power"] = adsrs["release_power"]
        # y["settings"]["lfos"] = x_init["settings"]["lfos"]

        del self.dict[CUSTOM_KEYS]

        with open(fname ,"w+") as f:
            json.dump(self.dict, f)