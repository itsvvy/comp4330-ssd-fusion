from pathlib import Path

import pytest
import torch

from layers.modules import MultiBoxLoss
from ssd import build_ssd


@pytest.mark.parametrize(
    "variant",
    ["baseline", "control", "fusion", "fusion_fixed", "fusion_active"],
)
def test_train_and_test_forward_shapes(variant):
    model = build_ssd("train", variant=variant)
    model.eval()
    with torch.no_grad():
        loc, conf, priors = model(torch.randn(1, 3, 300, 300))
    assert loc.shape == (1, 8732, 4)
    assert conf.shape == (1, 8732, 21)
    assert priors.shape == (8732, 4)

    detector = build_ssd("test", variant=variant)
    detector.eval()
    with torch.no_grad():
        detections = detector(torch.randn(1, 3, 300, 300))
    assert detections.shape == (1, 21, 200, 5)


def test_zero_gate_identity_and_branch_gradient():
    torch.manual_seed(7)
    baseline = build_ssd("train", variant="baseline").eval()
    fusion = build_ssd("train", variant="fusion").eval()
    incompatible = fusion.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert not incompatible.unexpected_keys
    assert all(
        key.startswith("fusion.") for key in incompatible.missing_keys
    )
    assert fusion.fusion.scale.item() == 0.0

    images = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        baseline_output = baseline(images)
        fusion_output = fusion(images)
    for left, right in zip(baseline_output, fusion_output):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)

    loss = sum(output.sum() for output in fusion(images))
    loss.backward()
    assert fusion.fusion.scale.grad is not None
    assert fusion.fusion.scale.grad.abs().item() > 0

    fusion.zero_grad(set_to_none=True)
    with torch.no_grad():
        fusion.fusion.scale.fill_(1.0)
    sum(output.sum() for output in fusion(images)).backward()
    assert fusion.fusion.project.weight.grad is not None
    assert fusion.fusion.pointwise.weight.grad is not None
    assert fusion.fusion.project.weight.grad.abs().sum().item() > 0


def test_fixed_fusion_identity_and_pointwise_gradient():
    torch.manual_seed(7)
    baseline = build_ssd("train", variant="baseline").eval()
    fusion = build_ssd("train", variant="fusion_fixed").eval()
    incompatible = fusion.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert not incompatible.unexpected_keys
    assert all(
        key.startswith("fusion.") for key in incompatible.missing_keys
    )
    assert fusion.fusion.scale.item() == 1.0

    images = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        baseline_output = baseline(images)
        fusion_output = fusion(images)
    for left, right in zip(baseline_output, fusion_output):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)

    loss = sum(output.sum() for output in fusion(images))
    loss.backward()
    assert fusion.fusion.pointwise.weight.grad is not None
    assert fusion.fusion.pointwise.weight.grad.abs().sum().item() > 0


def test_active_fusion_has_fixed_nonzero_scale():
    model = build_ssd("train", variant="fusion_active")
    assert model.fusion.scale.item() == 0.05
    assert not model.fusion.scale.requires_grad


def test_parameter_matched_control():
    fusion = build_ssd("train", variant="fusion")
    control = build_ssd("train", variant="control", control_channels=768)
    fusion_parameters = sum(
        parameter.numel() for parameter in fusion.fusion.parameters()
    )
    control_parameters = sum(
        parameter.numel() for parameter in control.control.parameters()
    )
    relative_error = abs(fusion_parameters - control_parameters) / fusion_parameters
    assert relative_error < 0.01


def test_baseline_checkpoint_only_allows_new_branch_keys(tmp_path: Path):
    baseline = build_ssd("train", variant="baseline")
    checkpoint = tmp_path / "baseline.pth"
    torch.save(baseline.state_dict(), checkpoint)

    fusion = build_ssd("train", variant="fusion")
    incompatible = fusion.load_weights(str(checkpoint))
    assert not incompatible.unexpected_keys
    assert all(
        key.startswith("fusion.") for key in incompatible.missing_keys
    )


def test_multibox_loss_backward_is_finite():
    model = build_ssd("train", variant="fusion")
    images = torch.randn(2, 3, 300, 300)
    targets = [
        torch.tensor(
            [[0.1, 0.1, 0.5, 0.5, 4], [0.5, 0.5, 0.9, 0.9, 14]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.2, 0.3, 0.4, 0.7, 6]],
            dtype=torch.float32,
        ),
    ]
    criterion = MultiBoxLoss(
        21,
        0.5,
        True,
        0,
        True,
        3,
        0.5,
        False,
        use_gpu=False,
    )
    predictions = model(images)
    loss_loc, loss_conf = criterion(predictions, targets)
    (loss_loc + loss_conf).backward()
    assert torch.isfinite(loss_loc)
    assert torch.isfinite(loss_conf)
    assert model.fusion.scale.grad is not None
