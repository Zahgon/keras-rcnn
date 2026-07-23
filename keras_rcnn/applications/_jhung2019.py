# -*- coding: utf-8 -*-

import numpy
import tensorflow

import keras_rcnn.models


class JHung2019(keras_rcnn.models.RCNN):
    def compile(self, optimizer, **kwargs):
        pass

    def predict(self, x, batch_size=None, verbose=0, steps=None, **kwargs):
        pass
