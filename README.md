![syntheon_logo](docs/syntheon-logo.png)

# Syntheon

Syntheon - [Pantheon](https://en.wikipedia.org/wiki/Pantheon,_Rome) for music synthesizers. 

Syntheon provides **parameter inference** for music synthesizers using *deep learning models*. Given an audio sample, Syntheon infers the best parameter preset for a given synthesizer that can recreate the audio sample. 

**Check out [this presentation](https://docs.google.com/presentation/d/1PA4fom6QvCW_YG8L0MMVumrAluljcymndNlaK2HW5t0/edit?usp=sharing) on the recent advances of synth parameter inference.

For now: 
- :heavy_check_mark: [Vital](https://vital.audio/) is supported
- [Dexed](https://asb2m10.github.io/dexed/) is work-in-progress

## Installation

```
python3 -m pip install syntheon
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

## Testing

```
python3 -m pytest
```

## Structure

For each synthesizer, we need to define:

- **converter** for preset format conversion: 
    - `serializeToDict`: convert preset file to a Python dictionary to be handled by inferencer
    - `parseToPluginFile`: convert Python dictionary back to preset file, to be loaded by the synthesizer

- **inferencer** for model inference:
    - `convert`: define the workflow of `load_model` -> `inference` -> `convert_to_preset`

## Contribution

Syntheon is actively under development, and contributions are welcomed. Some TODOs we have in mind include:

- Replicating state-of-the-art approaches
- Improving current model performance
- Incorporating new synthesizers 
- Code refactoring 😅
