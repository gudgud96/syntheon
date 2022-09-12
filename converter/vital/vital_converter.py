import json
import base64
import struct
from converter.converter import SynthConverter
from vital_constants import N_WAVETABLES, CUSTOM_KEYS
import numpy as np


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
                    
        except Exception as e:
            print(str(e))
        
        return self.dict
    
    def parseToPluginFile(self, fname):
        # encode custom part
        wavetables = self.dict[CUSTOM_KEYS]["wavetables"]
        for idx in range(len(N_WAVETABLES)):
            wavetable = wavetables[idx]["wavetable"]
            wavetable_name = wavetables[idx]["name"]
            wavetable_osc_level = wavetables[idx]["osc_level"]

            wavetable_str = self.base64_converter.encode(wavetable)
            self.dict["settings"]["wavetables"][idx]["groups"][0]["components"][0]["keyframes"][0]["wave_data"] = wavetable_str
            self.dict["settings"]["wavetables"][idx]["name"] = wavetable_name
            self.dict["settings"]["osc_{}_level".format(idx + 1)] = wavetable_osc_level
        
        # resume init settings for adsr filter
        # for idx in range(1, 7):
        #     y["settings"]["env_{}_attack".format(idx)] = x_init["settings"]["env_{}_attack".format(idx)]
        #     y["settings"]["env_{}_attack_power".format(idx)] = x_init["settings"]["env_{}_attack_power".format(idx)]
        #     y["settings"]["env_{}_decay".format(idx)] = x_init["settings"]["env_{}_decay".format(idx)]
        #     y["settings"]["env_{}_decay_power".format(idx)] = x_init["settings"]["env_{}_decay_power".format(idx)]
        #     y["settings"]["env_{}_delay".format(idx)] = x_init["settings"]["env_{}_delay".format(idx)]
        #     y["settings"]["env_{}_hold".format(idx)] = x_init["settings"]["env_{}_hold".format(idx)]
        #     y["settings"]["env_{}_release".format(idx)] = x_init["settings"]["env_{}_release".format(idx)]
        #     y["settings"]["env_{}_release_power".format(idx)] = x_init["settings"]["env_{}_release_power".format(idx)]
        #     y["settings"]["env_{}_sustain".format(idx)] = x_init["settings"]["env_{}_sustain".format(idx)]
        # y["settings"]["lfos"] = x_init["settings"]["lfos"]

        del self.dict[CUSTOM_KEYS]

        with open(fname ,"w+") as f:
            json.dump(self.dict, f)