# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend


def anchor(base_size=16, ratios=None, scales=None):
    pass


def bbox_transform(ex_rois, gt_rois):
    pass


def clip(boxes, shape):
    pass


def _mkanchors(ws, hs, x_ctr, y_ctr):
    pass


def _ratio_enum(anchor, ratios):
    pass


def _scale_enum(anchor, scales):
    pass


def _whctrs(anchor):
    pass


def shift(shape, stride, base_size=16, ratios=None, scales=None):
    pass


def intersection_over_union(output, target):
    pass


def smooth_l1(output, target, anchored=False, weights=None):
    pass


def focal_loss(target, output, gamma=2):
    pass


def softmax_classification(target, output, anchored=False, weights=None):
    pass


def bbox_transform_inv(boxes, deltas):
    pass
