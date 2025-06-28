#!/usr/bin/env python3
"""
Yolo class with output processing for object detection.
"""

import keras as K
import numpy as np


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
            anchors (numpy.ndarray): Anchor boxes with shape (outputs, anchor_boxes, 2).
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
            outputs (list): List of numpy.ndarrays containing predictions with shape
                           (grid_height, grid_width, anchor_boxes, 4 + 1 + classes).
            image_size (numpy.ndarray): Original image size [image_height, image_width].
            
        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                - boxes: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 4)
                - box_confidences: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 1)
                - box_class_probs: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, classes)
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
            box_y12 = (box_xy[..., 1:2] + grid_y) / grid_height
            
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
