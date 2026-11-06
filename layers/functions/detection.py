"""Post-processing for SSD inference outputs."""

import torch
from torch import nn

from data import voc as cfg
from ..box_utils import decode, nms


class Detect(nn.Module):
    """Decode detections and apply per-class non-maximum suppression."""

    def __init__(
        self,
        num_classes,
        bkg_label,
        top_k,
        conf_thresh,
        nms_thresh,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be positive")
        self.conf_thresh = conf_thresh
        self.variance = cfg["variance"]

    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = loc_data.new_zeros(
            num, self.num_classes, self.top_k, 5
        )
        conf_preds = conf_data.view(
            num, num_priors, self.num_classes
        ).transpose(2, 1)

        for image_index in range(num):
            decoded_boxes = decode(
                loc_data[image_index], prior_data, self.variance
            )
            conf_scores = conf_preds[image_index].clone()

            for class_index in range(1, self.num_classes):
                class_mask = conf_scores[class_index].gt(
                    self.conf_thresh
                )
                scores = conf_scores[class_index][class_mask]
                if scores.numel() == 0:
                    continue
                box_mask = class_mask.unsqueeze(1).expand_as(
                    decoded_boxes
                )
                boxes = decoded_boxes[box_mask].view(-1, 4)
                keep, count = nms(
                    boxes, scores, self.nms_thresh, self.top_k
                )
                output[image_index, class_index, :count] = torch.cat(
                    (
                        scores[keep[:count]].unsqueeze(1),
                        boxes[keep[:count]],
                    ),
                    1,
                )

        flat = output.contiguous().view(num, -1, 5)
        _, indices = flat[:, :, 0].sort(1, descending=True)
        _, rank = indices.sort(1)
        flat[(rank < self.top_k).unsqueeze(-1).expand_as(flat)].fill_(0)
        return output
