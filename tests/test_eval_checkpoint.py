import pytest
import torch

from eval import infer_control_channels


@pytest.mark.parametrize("wrapped", [False, True])
def test_infer_control_channels(tmp_path, wrapped):
    state_dict = {
        "control.pointwise.weight": torch.zeros(512, 766, 1, 1)
    }
    payload = {"model": state_dict} if wrapped else state_dict
    checkpoint = tmp_path / "control.pt"
    torch.save(payload, checkpoint)

    assert infer_control_channels(str(checkpoint)) == 766
