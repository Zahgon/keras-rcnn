# -*- coding: utf-8 -*-

import tensorflow


def resize(image, output_shape):
    pass


def transpose(x, axes=None):
    pass


def shuffle(x):
    pass


def gather_nd(params, indices):
    pass


def matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
):
    pass


def argsort(a):
    pass


def scatter_add_tensor(ref, indices, updates, name=None):
    pass


def meshgrid(*args, **kwargs):
    pass


newaxis = tensorflow.newaxis


def where(condition, x=None, y=None):
    pass


def non_maximum_suppression(boxes, scores, maximum, threshold=0.5):
    pass


def crop_and_resize(image, boxes, size):
    pass


def smooth_l1(output, target, anchored=False, weights=None):
    pass


def squeeze(a, axis=None):
    pass


def unique(x, return_index=False):
    pass


def pad(x, pad_width, mode):
    pass
