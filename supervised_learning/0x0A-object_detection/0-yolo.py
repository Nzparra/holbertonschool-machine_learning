#!/usr/bin/env python3
"""
    Yolo v3 algorithm to perform object detection
"""

import tensorflow.keras as K


class Yolo:
    """
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """constructor"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as fd:
            all_classes = fd.read()
            all_classes = all_classes.split('\n')
            if len(all_classes[-1]) == 0:
                all_classes = all_classes[:-1]
            self.class_names = all_classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
