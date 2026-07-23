# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend


class ObjectDetection(tensorflow.keras.layers.Layer):
    def __init__(self, padding=300, **kwargs):
        self.padding = padding

        super(ObjectDetection, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, training=None, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def compute_mask(self, inputs, mask=None):
        pass

    def detections(self, num_output, metadata, deltas, proposals, scores):
        pass

    @staticmethod
    def pad_bounding_boxes(x, padding):
        pass

    def get_config(self):
        pass
