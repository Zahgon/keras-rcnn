import io

import matplotlib.pyplot
import numpy
import tensorflow

import keras_rcnn.utils


def _generate_image(image, bounding_boxes):
    pass


class TensorBoard(tensorflow.keras.callbacks.TensorBoard):

    def __init__(self, generator):
        self.generator = generator

        super(TensorBoard, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        pass

    def _summarize_image(self):
        pass
