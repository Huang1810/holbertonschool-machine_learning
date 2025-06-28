#!/usr/bin/env python3
"""
YOLO v3 Object Detection - Extending with filter_boxes
"""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo instance.

        Args:
            model_path: path to Darknet Keras model
            classes_path: path to file with class names
            class_t: float, threshold for box score
            nms_t: float, IOU threshold for non-max suppression
            anchors: np.ndarray with shape (outputs, anchor_boxes, 2)
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().strip().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes model outputs.

        Args:
            outputs: list of np.ndarrays, model predictions
            image_size: np.ndarray (image height, image width)

        Returns:
            boxes, box_confidences, box_class_probs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])
            box_xy = self.sigmoid(output[..., 0:2])
            box_wh = np.exp(output[..., 2:4])
            anchors = self.anchors[i].reshape((1, 1, anchor_boxes, 2))
            box_wh *= anchors

            # Grid coordinates
            cx = np.tile(np.arange(0, grid_w), grid_h).reshape(grid_w, grid_h).T
            cy = np.tile(np.arange(0, grid_h), grid_w).reshape(grid_h, grid_w)

            cx = cx.reshape(grid_h, grid_w, 1, 1).repeat(anchor_boxes, axis=2)
            cy = cy.reshape(grid_h, grid_w, 1, 1).repeat(anchor_boxes, axis=2)
            grid = np.concatenate((cx, cy), axis=-1)

            box_xy += grid
            box_xy /= [grid_w, grid_h]
            box_wh /= self.model.input.shape[1:3]

            # Convert to (x1, y1, x2, y2)
            box_x1y1 = box_xy - (box_wh / 2)
            box_x2y2 = box_xy + (box_wh / 2)
            boxes_raw = np.concatenate((box_x1y1, box_x2y2), axis=-1)

            # Scale to image size
            boxes_raw[..., 0] *= image_width
            boxes_raw[..., 1] *= image_height
            boxes_raw[..., 2] *= image_width
            boxes_raw[..., 3] *= image_height

            boxes.append(boxes_raw)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters out boxes with low confidence scores.

        Args:
            boxes: list of np.ndarrays of shape (gh, gw, anchor, 4)
            box_confidences: list of np.ndarrays of shape (gh, gw, anchor, 1)
            box_class_probs: list of np.ndarrays of shape (gh, gw, anchor, classes)

        Returns:
            Tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_conf, box_prob in zip(boxes, box_confidences, box_class_probs):
            scores = box_conf * box_prob  # shape: (gh, gw, anchor, classes)
            class_scores = np.max(scores, axis=-1)
            class_ids = np.argmax(scores, axis=-1)

            # Create mask for boxes above threshold
            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(class_ids[mask])
            box_scores.append(class_scores[mask])

        if not filtered_boxes:
            return (np.array([]), np.array([]), np.array([]))

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
