# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend
import keras_rcnn.layers


class ObjectProposal(tensorflow.keras.layers.Layer):

    def __init__(self, maximum_proposals=300, minimum_size=16, stride=16, **kwargs):
        self.maximum_proposals = maximum_proposals

        self.minimum_size = minimum_size

        self.stride = stride

        super(ObjectProposal, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass


def filter_boxes(proposals, minimum):
    pass
