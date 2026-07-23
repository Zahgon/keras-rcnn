# -*- coding: utf-8 -*-

import tensorflow


class RCNNMaskLoss(tensorflow.keras.layers.Layer):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold

        super(RCNNMaskLoss, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        pass

    @staticmethod
    def intersection_over_union(a, b):
        pass

    @staticmethod
    def binary_crossentropy(_sentinel=None, target=None, output=None):
        pass

    @staticmethod
    def categorical_crossentropy(_sentinel=None, target=None, output=None):
        pass

    @staticmethod
    def compute_mask_loss(
        _sentinel=None,
        target_bounding_box=None,
        output_bounding_box=None,
        target_mask=None,
        output_mask=None,
        threshold=0.5,
    ):
        pass

    def compute_output_shape(self, input_shape):
        pass
