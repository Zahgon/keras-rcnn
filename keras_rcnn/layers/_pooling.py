# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend


class RegionOfInterest(tensorflow.keras.layers.Layer):

    def __init__(self, extent=(7, 7), strides=1, **kwargs):
        self.channels = None

        self.extent = extent

        self.stride = strides

        super(RegionOfInterest, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass


def log2_graph(x):
    pass


class RegionOfInterestAlignPyramid(tensorflow.keras.layers.Layer):
    def __init__(self, extent=(7, 7), strides=1, **kwargs):
        self.channels = None

        self.extent = extent

        self.stride = strides

        super(RegionOfInterestAlignPyramid, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass
