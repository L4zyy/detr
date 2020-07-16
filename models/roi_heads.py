# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from ..poolers import ROIPooler

class ROIHead(nn.Module):
    def __init__(self, box_in_features, box_pooler, box_head):
        """
        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
        """
        super().__init__()
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = nn.Linear(input_size, num_classes + 1)

    def forward(self, features, proposals):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        return predictions

def build_roi_head(args):
    in_features       = ['p2', 'p3', 'p4', 'p5']
    pooler_resolution = args.pooler_resolution
    pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
    sampling_ratio    = args.pooler_sampling_ratio
    pooler_type       = args.pooler_type

    # If StandardROIHeads is applied on multiple feature maps (as in FPN),
    # then we share the same predictors and therefore the channel counts must be the same
    in_channels = [input_shape[f].channels for f in in_features]
    # Check all channel counts are equal
    assert len(set(in_channels)) == 1, in_channels
    in_channels = in_channels[0]

    box_pooler = ROIPooler(
        output_size=pooler_resolution,
        scales=pooler_scales,
        sampling_ratio=sampling_ratio,
        pooler_type=pooler_type,
    )
    # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
    # They are used together so the "box predictor" layers should be part of the "box head".
    # New subclasses of ROIHeads do not need "box predictor"s.
    box_head = build_box_head(
        cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
    )

    model = ROIHead(in_features, box_pooler, box_head)

    return model