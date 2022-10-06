import os
from syntheon import infer_params


def test_dexed_inferencer():
    """
    just check if everything runs well for Dexed
    """
    output_params_file, eval_dict = infer_params(
        "test/test_audio/dexed_test_audio_1.wav", 
        "dexed", 
        enable_eval=True
    )
    assert os.path.exists(output_params_file)

    os.remove(output_params_file)


def test_vital_inferencer_1():
    """
    just check if everything runs well for Vital
    """
    loss_lst = [0.42, 0.11, 0.37, 0.06, 0.42, 0.18, 0.15]
    for idx in range(1, 8):
        output_params_file, eval_dict = infer_params(
            "test/test_audio/vital_test_audio_{}.wav".format(idx), 
            "vital", 
            enable_eval=True
        )
        assert os.path.exists(output_params_file)
        assert eval_dict["loss"] < loss_lst[idx - 1]
        os.remove(output_params_file)