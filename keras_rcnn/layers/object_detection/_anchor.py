# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend
import keras_rcnn.layers


class Anchor(tensorflow.keras.layers.Layer):
    def __init__(
        self,
        aspect_ratios=None,
        base_size=16,
        clobber_positives=False,
        negative_overlap=0.3,
        padding=0,
        positive_overlap=0.7,
        scales=None,
        stride=16,
        **kwargs
    ):
        if aspect_ratios is None:
            aspect_ratios = [0.5, 1, 2]  # [1:2, 1:1, 2:1]

        if scales is None:
            scales = [1, 2, 4, 8, 16]  # [32^{2}, 64^{2}, 128^{2}, 256^{2}, 512^{2}]

        self.padding = padding

        self.r = None
        self.c = None

        self.clobber_positives = clobber_positives

        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap

        self.stride = stride

        self.base_size = base_size

        self.aspect_ratios = tensorflow.keras.backend.variable(aspect_ratios)

        self.scales = tensorflow.keras.backend.variable(scales)

        self.__shifted_anchors = None

        self.metadata = None

        super(Anchor, self).__init__(**kwargs)

    @property
    def _shifted_anchors(self):
        pass

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def compute_mask(self, inputs, mask=None):
        pass

    def get_config(self):
        pass

    def _balance(self, labels):
        pass

    def _label(self, target, output, inds_inside):
        pass

    @staticmethod
    def _overlapping(output, target, inds_inside):
        pass

    @staticmethod
    def _subsample_negative_labels(labels, rpn_batchsize=256):
        pass

    @staticmethod
    def _subsample_positive_labels(labels, rpn_fg_fraction=0.5, rpn_batchsize=256):
        pass

    def _unmap(self, data, inds_inside, fill=0):
        pass

    def _inside_image(self, boxes):
        pass

    @staticmethod
    def _inside_and_outside_weights(
        anchors, subsample, positive_weight, proposed_inside_weights
    ):
        pass
