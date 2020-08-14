# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr_roi import build as roi_build


def build_model(args):
    if args.roi_head:
        return roi_build(args)
    return build(args)
