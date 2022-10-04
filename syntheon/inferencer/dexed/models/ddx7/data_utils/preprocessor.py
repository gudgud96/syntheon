import ddx7.core as core

class F0LoudnessRMSPreprocessor():
    """Scales 'f0_hz' and 'loudness_db' features."""
    def __init__(self):
        return

    def run(self,x):
        x['loudness_scaled'] = self.scale_db(x['loudness'])
        x['rms_scaled'] = self.scale_db(x['rms'])
        x['f0_scaled'] = self.scale_f0_hz(x['f0'])
        return x

    def scale_db(self,db):
        """Scales [-DB_RANGE, 0] to [0, 1]."""
        return (db / core._DB_RANGE) + 1.0

    def scale_f0_hz(self,f0_hz):
        """Scales [0, Nyquist] Hz to [0, 1.0] MIDI-scaled."""
        return core.hz_to_midi(f0_hz) / core._F0_RANGE