"""VGG16-SSD300 with an optional lightweight cross-scale branch."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import coco, voc
from layers import Detect, L2Norm, PriorBox


class GatedFeatureBranch(nn.Module):
    """Project a source feature map and add it through a zero gate.

    ``fusion`` reads conv7 at 19x19 and upsamples to conv4_3 resolution.
    ``control`` reads conv4_3 at 38x38 and therefore has no cross-scale path.
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels=512,
        upsample=False,
    ):
        super().__init__()
        self.upsample = upsample
        self.project = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1, bias=False
        )
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            hidden_channels, output_channels, kernel_size=1, bias=False
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, shallow, deep=None):
        source = deep if deep is not None else shallow
        branch = F.relu(self.project(source), inplace=True)
        if self.upsample:
            branch = F.interpolate(
                branch,
                size=shallow.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        branch = F.relu(self.depthwise(branch), inplace=True)
        branch = self.pointwise(branch)
        return shallow + self.scale * branch


class HighResolutionFusion(nn.Module):
    """Fuse conv3_3, conv4_3 and conv7 at the conv3_3 resolution."""

    def __init__(
        self,
        input_channels=(256, 512, 1024),
        project_channels=64,
        output_channels=512,
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(channels, project_channels, kernel_size=1)
                for channels in input_channels
            ]
        )
        self.fusion = nn.Conv2d(
            project_channels * len(input_channels),
            output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, high, shallow, deep):
        target_size = high.shape[-2:]
        projected = []
        for source, projection in zip(
            (high, shallow, deep), self.projections
        ):
            feature = F.relu(projection(source), inplace=True)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature)
        return F.relu(
            self.fusion(torch.cat(projected, dim=1)), inplace=True
        )


class SSD(nn.Module):
    """SSD300 with optional fusion or parameter-matched control branch."""

    def __init__(
        self,
        phase,
        size,
        base,
        extras,
        head,
        num_classes,
        variant="baseline",
        control_channels=768,
    ):
        super().__init__()
        if phase not in {"train", "test"}:
            raise ValueError(f"unknown phase: {phase}")
        if variant not in {
            "baseline",
            "control",
            "fusion",
            "small_head_fusion",
        }:
            raise ValueError(f"unknown variant: {variant}")

        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        self.variant = variant
        self.cfg = (coco, voc)[num_classes == 21]
        self.register_buffer(
            "priors", PriorBox(self.cfg).forward(), persistent=False
        )

        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if variant == "fusion":
            self.fusion = GatedFeatureBranch(
                1024, 512, output_channels=512, upsample=True
            )
        elif variant == "control":
            if control_channels <= 0:
                raise ValueError("control_channels must be positive")
            self.control = GatedFeatureBranch(
                512,
                control_channels,
                output_channels=512,
                upsample=False,
            )
        elif variant == "small_head_fusion":
            self.small_fusion = HighResolutionFusion()
            self.small_loc = nn.Conv2d(
                512, 4 * 4, kernel_size=3, padding=1
            )
            self.small_conf = nn.Conv2d(
                512, 4 * num_classes, kernel_size=3, padding=1
            )
            self.small_loc.load_state_dict(self.loc[0].state_dict())
            self.small_conf.load_state_dict(self.conf[0].state_dict())
            small_cfg = {
                "min_dim": 300,
                "feature_maps": [75],
                "min_sizes": [12],
                "max_sizes": [24],
                "steps": [4],
                "aspect_ratios": [[2]],
                "variance": cfg["variance"],
                "clip": True,
                "name": cfg["name"],
            }
            self.priors = torch.cat(
                (self.priors, PriorBox(small_cfg).forward()), dim=0
            )
        elif phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        # Detector also needs to work in test phase for all variants.
        if phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def _base_features(self, x):
        high_res = None
        for index in range(23):
            x = self.vgg[index](x)
            if index == 15:
                high_res = x
        shallow_raw = x
        shallow = self.L2Norm(shallow_raw)
        for index in range(23, len(self.vgg)):
            x = self.vgg[index](x)
        return shallow, x, high_res

    def forward(self, x):
        shallow, deep, high_res = self._base_features(x)
        if self.variant == "fusion":
            shallow = self.fusion(shallow, deep)
        elif self.variant == "control":
            shallow = self.control(shallow)

        sources = [shallow, deep]
        features = deep
        for index, layer in enumerate(self.extras):
            features = F.relu(layer(features), inplace=True)
            if index % 2 == 1:
                sources.append(features)

        locations = []
        confidences = []
        for source, loc_layer, conf_layer in zip(
            sources, self.loc, self.conf
        ):
            locations.append(
                loc_layer(source)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            confidences.append(
                conf_layer(source)
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        if self.variant == "small_head_fusion":
            small_feature = self.small_fusion(high_res, shallow, deep)
            locations.append(
                self.small_loc(small_feature)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            confidences.append(
                self.small_conf(small_feature)
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        locations = torch.cat(
            [item.view(item.size(0), -1) for item in locations], 1
        )
        confidences = torch.cat(
            [item.view(item.size(0), -1) for item in confidences], 1
        )

        if self.phase == "test":
            return self.detect(
                locations.view(locations.size(0), -1, 4),
                self.softmax(
                    confidences.view(
                        confidences.size(0), -1, self.num_classes
                    )
                ),
                self.priors.to(x.device),
            )

        return (
            locations.view(locations.size(0), -1, 4),
            confidences.view(
                confidences.size(0), -1, self.num_classes
            ),
            self.priors.to(x.device),
        )

    def load_weights(self, base_file):
        checkpoint = torch.load(
            base_file, map_location="cpu", weights_only=False
        )
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        incompatible = self.load_state_dict(checkpoint, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        allowed_prefix = {
            "fusion": "fusion.",
            "control": "control.",
            "small_head_fusion": (
                "small_fusion.",
                "small_loc.",
                "small_conf.",
            ),
        }.get(self.variant)

        def allowed_missing(key):
            if allowed_prefix is None:
                return False
            prefixes = (
                allowed_prefix
                if isinstance(allowed_prefix, tuple)
                else (allowed_prefix,)
            )
            return any(key.startswith(prefix) for prefix in prefixes)

        invalid_missing = [
            key for key in missing if not allowed_missing(key)
        ]
        if unexpected or invalid_missing:
            raise RuntimeError(
                "checkpoint mismatch: "
                f"unexpected={unexpected}, missing={invalid_missing}"
            )
        return incompatible


def vgg(cfg, in_channels, batch_norm=False):
    layers = []
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif value == "C":
            layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            )
        else:
            conv = nn.Conv2d(
                in_channels, value, kernel_size=3, padding=1
            )
            if batch_norm:
                layers.extend(
                    [conv, nn.BatchNorm2d(value), nn.ReLU(inplace=True)]
                )
            else:
                layers.extend([conv, nn.ReLU(inplace=True)])
            in_channels = value

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers.extend(
        [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    )
    return layers


def add_extras(cfg, in_channels, batch_norm=False):
    layers = []
    flag = False
    for index, value in enumerate(cfg):
        if in_channels != "S":
            if value == "S":
                kernel = (1, 3)[flag]
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        cfg[index + 1],
                        kernel_size=kernel,
                        stride=2,
                        padding=1,
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        value,
                        kernel_size=(1, 3)[flag],
                    )
                )
            flag = not flag
        in_channels = value
    return layers


def multibox(vgg_layers, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for index, source_index in enumerate([21, -2]):
        channels = vgg_layers[source_index].out_channels
        loc_layers.append(
            nn.Conv2d(
                channels, cfg[index] * 4, kernel_size=3, padding=1
            )
        )
        conf_layers.append(
            nn.Conv2d(
                channels,
                cfg[index] * num_classes,
                kernel_size=3,
                padding=1,
            )
        )
    for offset, layer in enumerate(extra_layers[1::2], start=2):
        loc_layers.append(
            nn.Conv2d(
                layer.out_channels,
                cfg[offset] * 4,
                kernel_size=3,
                padding=1,
            )
        )
        conf_layers.append(
            nn.Conv2d(
                layer.out_channels,
                cfg[offset] * num_classes,
                kernel_size=3,
                padding=1,
            )
        )
    return vgg_layers, extra_layers, (loc_layers, conf_layers)


base = {
    "300": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "C",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    "512": [],
}
extras = {
    "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
    "512": [],
}
mbox = {
    "300": [4, 6, 6, 6, 4, 4],
    "512": [],
}


def build_ssd(
    phase,
    size=300,
    num_classes=21,
    variant="baseline",
    control_channels=768,
):
    if size != 300:
        raise ValueError(f"only SSD300 is supported, got {size}")
    base_layers, extra_layers, head = multibox(
        vgg(base[str(size)], 3),
        add_extras(extras[str(size)], 1024),
        mbox[str(size)],
        num_classes,
    )
    return SSD(
        phase,
        size,
        base_layers,
        extra_layers,
        head,
        num_classes,
        variant=variant,
        control_channels=control_channels,
    )
