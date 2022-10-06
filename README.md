# Syntheon

Syntheon - the Pantheon (temple of gods) for music synthesizers.

DX7 patches: https://yamahablackboxes.com/collection/yamaha-dx7-synthesizer/patches/

## Testing

```
python3 -m inferencer.vital.vital_inferencer

python3 -m inferencer.dexed.dexed_inferencer

python3 -m pytest

```

## Usage

```python
from syntheon import infer_params

output_params_file, eval_dict = infer_params(
    "your_audio.wav", 
    "vital", 
    enable_eval=True
)
```