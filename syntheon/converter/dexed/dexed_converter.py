from syntheon.converter.converter import SynthConverter
import mido
from pathlib import Path
from syntheon.converter.dexed.dexed_constants import voice_struct, VOICE_PARAMETER_RANGES, header_struct,\
    header_bytes, voice_bytes, N_VOICES, N_OSC, KEYS


def take(take_from, n):
    for _ in range(n):
        yield next(take_from)


def checksum(data):
    return (128-sum(data)&127)%128


class DexedConverter(SynthConverter):
    def __init__(self):
        SynthConverter.__init__(self)
        self.keys = KEYS

    def serializeToDict(self, fname):
        path = Path(fname).expanduser()
        try:
            preset = mido.read_syx_file(path.as_posix())[0]
        except IndexError as e:
            return None
        except ValueError as e:
            return None
        if len(preset.data) == 0:
            return None

        def get_voice(data):
            unpacked = voice_struct.unpack(data)
            # TODO: need to take actions after verify, skip for now
            # self.verify(unpacked, VOICE_PARAMETER_RANGES)
            return unpacked

        get_header = header_struct.unpack
        sysex_iter = iter(preset.data)
        lst = []
        try:
            header = get_header(bytes(take(sysex_iter, len(header_bytes))))
            for idx in range(N_VOICES):
                x = get_voice(bytes(take(sysex_iter, len(voice_bytes)))) 
                lst.append(x)
            
            self.dict = lst
            return lst
        except RuntimeError:
            return None
    
    def parseToPluginFile(self, fname):
        def encode_head():
            header = [  '0x43',
                        '0x00',
                        '0x09',
                        '0x20',
                        '0x00',]

            return [int(i, 0) for i in header]

        def encode_osc(params, n):
            oscillator_params = []

            oscillator_params += [params[f'{n}_R1']]
            oscillator_params += [params[f'{n}_R2']]
            oscillator_params += [params[f'{n}_R3']]
            oscillator_params += [params[f'{n}_R4']]
            oscillator_params += [params[f'{n}_L1']]
            oscillator_params += [params[f'{n}_L2']]
            oscillator_params += [params[f'{n}_L3']]
            oscillator_params += [params[f'{n}_L4']]
            oscillator_params += [params[f'{n}_BP']]
            oscillator_params += [params[f'{n}_LD']]
            oscillator_params += [params[f'{n}_RD']]

            RC = params[f'{n}_RC'] << 2
            LC = params[f'{n}_LC']
            oscillator_params += [RC | LC]

            DET = params[f'{n}_DET'] << 3
            RS = params[f'{n}_RS']
            oscillator_params += [DET | RS]

            KVS = params[f'{n}_KVS'] << 2
            AMS = params[f'{n}_AMS'] 
            oscillator_params += [KVS|AMS]
            oscillator_params += [params[f'{n}_OL']]

            FC = params[f'{n}_FC'] << 1
            M = params[f'{n}_M']
            oscillator_params += [FC|M]
            oscillator_params += [params[f'{n}_FF']]

            return oscillator_params

        def encode_global(params):
            global_params = []
            global_params += [params['PR1']]
            global_params += [params['PR2']]
            global_params += [params['PR3']]
            global_params += [params['PR4']]
            global_params += [params['PL1']]
            global_params += [params['PL2']]
            global_params += [params['PL3']]
            global_params += [params['PL4']]

            global_params += [params['ALG']]

            OKS = params['OKS'] << 3
            FB = params['FB']

            global_params += [OKS|FB]
            global_params += [params['LFS']]
            global_params += [params['LFD']]
            global_params += [params['LPMD']]
            global_params += [params['LAMD']]

            LPMS = params['LPMS'] << 4
            LFW = params['LFW'] << 1
            LKS = params['LKS']
            global_params += [LPMS | LFW | LKS]
            global_params += [params['TRNSP']]
            global_params += [params[f'NAME CHAR {i + 1}'] for i in range(10)]

            return global_params

        try:
            head = encode_head()

            data = []
            assert len(self.dict) == N_VOICES

            # voices
            last_params = None
            for params in self.dict:
                if len(params.keys()) == 0:
                    params = last_params
                else:
                    last_params = params
                for osc in range(N_OSC):
                    data += encode_osc(params, osc)

                data += encode_global(params)


            this_checksum = checksum(data)
            output = [*head, *data, this_checksum]
            
            message = mido.Message('sysex', data=output)
            mido.write_syx_file(fname, [message])
            return 0
        
        except Exception as e:
            print(str(e))
            return -1
    
    def verify(self, actual, ranges):
        super().verify()
        assert set(actual.keys())==set(ranges.keys()), 'Params dont match'
        for key in actual:
            if not actual[key] in ranges[key]:
                print("returning false", key, actual[key])
                return False
        return True


if __name__ == "__main__":
    dx_converter = DexedConverter()
    dx_converter.serializeToDict("Dexed_01.syx")
    dx_converter.printMessage()
    dx_converter.parseToPluginFile("testing.syx")



