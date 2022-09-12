class SynthConverter:
    
    def __init__(self):
        self.dict = None
        self.keys = []      # keys that need to be filled for this synth
    
    def serializeToDict(self, fname):
        """
        From plugin file to protobuf.

        Args:
        fname - input file name
        """
        return None
    
    def parseToPluginFile(self, fname):
        """
        From protobuf to plugin file.

        Args:
        fname - output file name
        """
        return None
    
    def printMessage(self):
        """
        Print synth parameters.
        """
        if self.dict:
            print(self.dict)
        
        else:
            raise ValueError("synth parameters not serialized yet")
    
    def keys(self):
        return self.keys
    
    def verify(self):
        """
        Verify if params are valid. Used in serializeToDict method.
        """
        if self.dict is None:
            raise ValueError("synth parameters not serialized yet")
        
        # value range checks can leave to derived classes
        for key in self.keys:
            if isinstance(self.dict, list):
                for elem in self.dict:
                    if key not in elem:
                        raise ValueError("specified key not in synth parameters: {}".format(key))
            elif isinstance(self.dict, dict):
                if key not in elem:
                        raise ValueError("specified key not in synth parameters: {}".format(key))