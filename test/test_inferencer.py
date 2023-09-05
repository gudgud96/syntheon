import os
import glob
from syntheon import infer_params


# def test_dexed_inferencer():
#     """
#     just check if everything runs well for Dexed
#     """
#     output_params_file, eval_dict = infer_params(
#         "test/test_audio/dexed_test_audio_1.wav", 
#         "dexed", 
#         enable_eval=True
#     )
#     assert os.path.exists(output_params_file)

#     os.remove(output_params_file)


def test_vital_inferencer_1():
    """
    just check if everything runs well for Vital
    """
    loss_lst = [0.11, 0.06, 0.37, 0.42, 0.18, 0.15]
    audios = sorted(glob.glob("test/test_audio/vital_*.wav"))
    for i in range(len(audios)):
        output_params_file, eval_dict = infer_params(
            audios[i],
            "vital", 
            enable_eval=True
        )
        assert os.path.exists(output_params_file)
        assert eval_dict["loss"] < loss_lst[i]
        os.remove(output_params_file)