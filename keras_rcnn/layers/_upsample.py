# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend


class Upsample(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Upsample, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass
