# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend


class ProposalTarget(tensorflow.keras.layers.Layer):

    def __init__(
        self,
        foreground=0.5,
        foreground_threshold=(0.5, 1.0),
        background_threshold=(0.1, 0.5),
        maximum_proposals=32,
        **kwargs
    ):
        """
        :param foreground:
        :param foreground_threshold:
        :param background_threshold:
        :param maximum_proposals:
        """
        self._batch_size = None

        self.foreground = foreground

        self.foreground_threshold = foreground_threshold
        self.background_threshold = background_threshold

        self.maximum_proposals = maximum_proposals

        self.rois_per_image = self.maximum_proposals / self.batch_size
        self.fg_rois_per_image = tensorflow.keras.backend.cast(
            self.foreground * self.rois_per_image, "int32"
        )

        self.fg_rois_per_this_image = None

        super(ProposalTarget, self).__init__(**kwargs)

    @property
    def batch_size(self):
        pass

    @batch_size.setter
    def batch_size(self, x):
        pass

    def build(self, input_shape):
        pass

    def call(self, inputs, training=None):
        pass

    def get_config(self):
        pass

    def sample(self, proposals, true_bounding_boxes, true_labels):
        pass

    def set_label_background(self, labels):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def compute_mask(self, inputs, mask=None):
        pass

    def get_bbox_targets(self, rois, gt_boxes, labels, num_classes):
        pass

    def find_foreground_and_background_proposal_indices(self, max_overlaps):
        pass

    def sample_indices(self, indices, size):
        pass

    def get_bbox_regression_labels(self, bbox_target_data, labels, num_classes):
        pass
