# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.models


class Hollandi2019(keras_rcnn.models.MaskRCNN):
    def compile(self, optimizer, **kwargs):
        pass

    def predict(self, x, batch_size=None, verbose=0, steps=None, **kwargs):
        pass
