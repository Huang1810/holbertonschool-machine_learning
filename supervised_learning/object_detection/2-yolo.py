#!/usr/bin/env python3
"""
YOLO v3 - Initialize and process model outputs for object detection
"""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """YOLO v3 object detection class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the YOLO object detector
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = file.read().strip().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process model outputs into bounding boxes
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            box_conf = self.sigmoid(output[..., 4:5])
            class_probs = self.sigmoid(output[..., 5:])
            box_xy = self.sigmoid(output[..., 0:2])
            box_wh = np.exp(output[..., 2:4])
            anchors = self.anchors[i].reshape(
                (1, 1, len(self.anchors[i]), 2)
            )
            box_wh *= anchors

            col = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w)
            row = np.tile(np.arange(grid_h), grid_w).reshape(grid_w, grid_h).T

            col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
            row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)

            box_xy += np.concatenate((col, row), axis=-1)
            box_xy /= (grid_w, grid_h)
            box_wh /= (self.model.input.shape[1],
                       self.model.input.shape[2])

            box_xy -= (box_wh / 2)
            box_coords = np.concatenate((box_xy, box_xy + box_wh), axis=-1)

            box_coords[..., 0] *= image_width
            box_coords[..., 1] *= image_height
            box_coords[..., 2] *= image_width
            box_coords[..., 3] *= image_height

            boxes.append(box_coords)
            box_confidences.append(box_conf)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes using class score threshold
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, class_prob in zip(boxes,
                                         box_confidences,
                                         box_class_probs):
            scores = conf * class_prob
            max_scores = np.max(scores, axis=-1)
            max_classes = np.argmax(scores, axis=-1)

            mask = max_scores >= self.class_t
            filtered_boxes.append(box[mask])
            box_classes.append(max_classes[mask])
            box_scores.append(max_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores
