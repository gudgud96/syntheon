import os
from syntheon import infer_params


def test_dexed_inferencer():
    """
    just check if everything runs well for Dexed
    """
    output_params_file = infer_params("test/test_audio/dexed_test_audio_1.wav", "dexed")
    assert os.path.exists(output_params_file)

    os.remove(output_params_file)


def test_vital_inferencer():
    """
    just check if everything runs well for Vital
    """
    output_params_file = infer_params("test/test_audio/vital_test_audio_1.wav", "vital")
    assert os.path.exists(output_params_file)

    os.remove(output_params_file)