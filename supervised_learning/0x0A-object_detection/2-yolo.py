#!/usr/bin/env python3
"""
    Yolo v3 algorithm to perform object detection
"""

import numpy as np
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

    def process_outputs(self, outputs, image_size):
        """
        Returns a tuple of (boxes, box_confidences, box_class_probs):
        """
        final_boxes = []
        confidence_boxes = []
        prop_boxes = []
        im_h, im_w = image_size
        for index, output in enumerate(outputs):
            grid_h, grid_w, anchor, total = output.shape
            t_prediction = output[:, :, :, :4]
            boxes = np.zeros(t_prediction.shape)
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            pw = self.anchors[:, :, 0]
            ph = self.anchors[:, :, 1]
            pw_actual = pw[index].reshape(1, 1, len(pw[index]))
            ph_actual = ph[index].reshape(1, 1, len(ph[index]))
            cx = np.tile(np.arange(0, grid_w), grid_h)
            cx = cx.reshape(grid_w, grid_w, 1)
            cy = np.tile(np.arange(0, grid_w), grid_h)
            cy = (cy.reshape(grid_h, grid_h).T).reshape(grid_h, grid_h, 1)
            bx = (1 / (1 + np.exp(-t_x))) + cx
            by = (1 / (1 + np.exp(-t_y))) + cy
            bw = np.exp(t_w) * pw_actual
            bh = np.exp(t_h) * ph_actual
            bx = bx / grid_w
            by = by / grid_h
            bw = bw / self.model.input.shape[1].value
            bh = bh / self.model.input.shape[2].value
            x1 = (bx - (bw / 2)) * im_w
            y1 = (by - (bh / 2)) * im_h
            x2 = (bx + (bw / 2)) * im_w
            y2 = (by + (bh / 2)) * im_h
            boxes[:, :, :, 0] = x1
            boxes[:, :, :, 1] = y1
            boxes[:, :, :, 2] = x2
            boxes[:, :, :, 3] = y2
            final_boxes.append(boxes)
            t_c = output[:, :, :, 4]
            confidence = (1 / (1 + np.exp(-t_c)))
            confidence = confidence.reshape(grid_h, grid_w, anchor, 1)
            confidence_boxes.append(confidence)
            t_cprops = output[:, :, :, 5:]
            class_props = (1 / (1 + np.exp(-t_cprops)))
            prop_boxes.append(class_props)
        return (final_boxes, confidence_boxes, prop_boxes)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Returns a tuple of (filtered_boxes, box_classes, box_scores)"""
        multi = []
        for confidence, clase in zip(box_confidences, box_class_probs):
            multi.append(confidence * clase)
        index_class = [np.argmax(elem, axis=-1) for elem in multi]
        index_class = [elem.reshape(-1) for elem in index_class]
        index_class = np.concatenate(index_class)
        score_class = [np.max(elem, axis=-1) for elem in multi]
        score_class = [elem.reshape(-1) for elem in score_class]
        score_class = np.concatenate(score_class)
        mask = np.where(score_class >= self.class_t)
        box_class = index_class[mask]
        box_score = score_class[mask]
        filter_box = [elem.reshape(-1, 4) for elem in boxes]
        filter_box = np.concatenate(filter_box)
        filter_box = filter_box[mask]
        return (filter_box, box_class, box_score)
