# -*- coding: utf-8 -*-

import numpy
import skimage.color
import skimage.exposure
import skimage.io
import skimage.transform
import tensorflow


class BoundingBoxException(Exception):
    pass


class MissingImageException(Exception):
    pass


class DictionaryIterator(tensorflow.keras.preprocessing.image.Iterator):
    def __init__(
        self,
        dictionary,
        categories,
        target_size,
        generator,
        batch_size=1,
        color_mode="rgb",
        data_format=None,
        mask_size=(28, 28),
        seed=None,
        shuffle=False,
    ):
        if color_mode not in {"grayscale", "rgb"}:
            raise ValueError

        self.batch_size = batch_size

        self.categories = categories

        if color_mode == "rgb":
            self.channels = 3
        else:
            self.channels = 1

        self.color_mode = color_mode

        if data_format is None:
            data_format = tensorflow.keras.backend.image_data_format()

        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError

        self.data_format = data_format

        self.dictionary = dictionary

        self.generator = generator

        if self.color_mode == "grayscale":
            if self.data_format == "channels_first":
                self.image_shape = (*target_size, 1)
            else:
                self.image_shape = (1, *target_size)
        else:
            if self.data_format == "channels_last":
                self.image_shape = (*target_size, 3)
            else:
                self.image_shape = (3, *target_size)

        self.mask_size = mask_size

        self.maximum = numpy.max(target_size)

        self.minimum = numpy.min(target_size)

        self.n_categories = len(self.categories) + 1

        self.n_samples = len(self.dictionary)

        self.target_size = target_size

        super(DictionaryIterator, self).__init__(
            self.n_samples, batch_size, shuffle, seed
        )

    def next(self):
        pass

    def find_scale(self, image):
        pass

    def _clear_border(self, bounding_boxes):
        pass

    @staticmethod
    def _crop_bounding_boxes(bounding_boxes, boundary):
        pass

    def _crop_image(self, image):
        pass

    def _get_batches_of_transformed_samples(self, selection):
        pass

    def _transform_samples(self, batch_index, image_index):
        pass

    @staticmethod
    def _cropped_objects(x_bounding_boxes):
        pass

    def _shuffle_objects(self, x_bounding_boxes, x_categories, x_masks):
        pass


class ObjectDetectionGenerator:
    def __init__(
        self,
        clear_border=False,
        crop_size=None,
        data_format=None,
        horizontal_flip=False,
        preprocessing_function=None,
        rescale=False,
        rotation_range=0.0,
        samplewise_center=False,
        vertical_flip=False,
    ):
        self.clear_border = clear_border

        self.crop_size = crop_size

        self.data_format = data_format

        self.horizontal_flip = horizontal_flip

        self.preprocessing_function = preprocessing_function

        self.rescale = rescale

        self.rotation_range = rotation_range

        self.samplewise_center = samplewise_center

        self.vertical_flip = vertical_flip

    def flow_from_dictionary(
        self,
        dictionary,
        categories,
        target_size,
        batch_size=1,
        color_mode="rgb",
        data_format=None,
        mask_size=(28, 28),
        shuffle=True,
        seed=None,
    ):
        pass

    def standardize(self, image):
        pass
