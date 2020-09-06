import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from util import box_ops

import fvcore.nn.weight_init as weight_init

from detectron2.layers import ShapeSpec, Conv2d, Linear, get_norm
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.structures import Boxes

class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

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

        self.box_pred_input_size = 1024
        self.num_classes = 91
        self.box_predictor = nn.Linear(self.box_pred_input_size, self.num_classes + 1)

    def forward(self, features, boxes):
        b, q, _ = boxes.shape
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [Boxes(box_ops.box_cxcywh_to_xyxy(x)) for x in boxes])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        return predictions.view(b, q, self.num_classes+1)

def build_roi_head(args):
    num_channels = 256
    in_features       = ['p2', 'p3', 'p5']
    input_shape = {
        'p2': ShapeSpec(channels=num_channels, stride=4),
        'p3': ShapeSpec(channels=num_channels, stride=8),
        'p5': ShapeSpec(channels=num_channels, stride=16),
        }

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

    num_conv = 4
    conv_dim = 256
    num_fc = 1
    fc_dim = 1024

    box_head = FastRCNNConvFCHead(
        ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution),
        conv_dims=[conv_dim]*num_conv,
        fc_dims=[fc_dim]*num_fc,
        conv_norm=""
    )

    model = ROIHead(in_features, box_pooler, box_head)

    return model