#!/usr/bin/env python3
"""
Yolo class with methods for loading images and processing outputs
for object detection.
"""

import keras as K
import numpy as np
import cv2
import os


class Yolo:
    """
    Yolo class for object detection using YOLOv3 algorithm.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo class with the specified parameters.

        Args:
            model_path (str): Path to the Darknet Keras model.
            classes_path (str): Path to the file containing class names.
            class_t (float): Box score threshold for initial filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes with shape
                (outputs, anchor_boxes, 2).
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_class_names(self, classes_path):
        """
        Load class names from a file.

        Args:
            classes_path (str): Path to the file containing class names.

        Returns:
            list: List of class names.
        """
        with open(classes_path, 'r') as file:
            class_names = file.read().strip().split('\n')
        return class_names

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs to bounding boxes.

        Args:
            outputs (list): List of numpy.ndarrays containing predictions with
                shape (grid_height, grid_width, anchor_boxes, 4 + 1 + classes).
            image_size (numpy.ndarray): Original image size [image_height,
                image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                - boxes: List of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, 4)
                - box_confidences: List of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, 1)
                - box_class_probs: List of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, classes)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for output in outputs:
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box coordinates, confidence, and class probabilities
            box_xy = output[..., :2]
            box_wh = output[..., 2:4]
            box_confidence = output[..., 4:5]
            box_class_prob = output[..., 5:]

            # Apply sigmoid activation
            box_xy = 1 / (1 + np.exp(-box_xy))
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))

            # Get grid dimensions
            grid_x = np.tile(np.arange(grid_width), (grid_height, 1))
            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_x = np.tile(grid_x, (1, 1, anchor_boxes))

            grid_y = np.tile(np.arange(grid_height), (grid_width, 1)).T
            grid_y = np.expand_dims(grid_y, axis=-1)
            grid_y = np.tile(grid_y, (1, 1, anchor_boxes))

            # Calculate box coordinates
            box_x = (box_xy[..., 0:1] + grid_x) / grid_width
            box_y = (box_xy[..., 1:2] + grid_y) / grid_height

            # Get appropriate anchors
            stride = input_width // grid_width
            anchor_idx = len(outputs) - (input_width // stride // 13)
            anchors = self.anchors[anchor_idx]

            # Calculate box width and height
            box_w = np.exp(box_wh[..., 0:1]) * anchors[:, 0:1] / input_width
            box_h = np.exp(box_wh[..., 1:2]) * anchors[:, 1:2] / input_height

            # Convert to corner coordinates
            box_x1 = (box_x - box_w / 2) * image_width
            box_y1 = (box_y - box_h / 2) * image_height
            box_x2 = (box_x + box_w / 2) * image_width
            box_y2 = (box_y + box_h / 2) * image_height

            box = np.concatenate([box_x1, box_y1, box_x2, box_y2], axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on confidence threshold.

        Args:
            boxes (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4).
            box_confidences (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1).
            box_class_probs (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes).

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                - filtered_boxes: numpy.ndarray of shape (?, 4) containing
                  filtered bounding boxes.
                - box_classes: numpy.ndarray of shape (?) containing class indices.
                - box_scores: numpy.ndarray of shape (?) containing box scores.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_prob in zip(boxes,
                                               box_confidences,
                                               box_class_probs):
            # Calculate box scores
            scores = confidence * class_prob
            max_scores = np.max(scores, axis=-1)
            max_classes = np.argmax(scores, axis=-1)

            # Apply threshold
            mask = max_scores >= self.class_t
            filtered_boxes.append(box[mask])
            box_classes.append(max_classes[mask])
            box_scores.append(max_scores[mask])

        # Concatenate results
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-max Suppression to filtered boxes.

        Args:
            filtered_boxes (numpy.ndarray): Shape (?, 4) containing filtered
                bounding boxes.
            box_classes (numpy.ndarray): Shape (?) containing class indices.
            box_scores (numpy.ndarray): Shape (?) containing box scores.

        Returns:
            tuple: (box_predictions, predicted_box_classes, predicted_box_scores)
                - box_predictions: numpy.ndarray of shape (?, 4) containing
                  predicted bounding boxes.
                - predicted_box_classes: numpy.ndarray of shape (?) containing
                  predicted class indices.
                - predicted_box_scores: numpy.ndarray of shape (?) containing
                  predicted box scores.
        """
        unique_classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            # Get boxes for current class
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            # Sort by scores
            idx = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[idx]
            cls_scores = cls_scores[idx]

            while len(cls_boxes) > 0:
                # Add top box to predictions
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                # Calculate IoU
                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                w = np.maximum(0, x2 - x1)
                h = np.maximum(0, y2 - y1)
                inter = w * h

                area1 = (cls_boxes[0, 2] - cls_boxes[0, 0]) * \
                        (cls_boxes[0, 3] - cls_boxes[0, 1])
                area2 = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * \
                        (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                union = area1 + area2 - inter

                iou = inter / union

                # Remove boxes with IoU above threshold
                mask = iou < self.nms_t
                cls_boxes = cls_boxes[1:][mask]
                cls_scores = cls_scores[1:][mask]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    def load_images(self, folder_path):
        """
        Load images from a folder path.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            list: List of images as numpy.ndarrays.
            list: List of image file paths.
        """
        images = []
        image_paths = []

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(folder_path, filename)
                img = cv2.imread(full_path)
                if img is not None:
                    images.append(img)
                    image_paths.append(full_path)

        return images, image_paths
