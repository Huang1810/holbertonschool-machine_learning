#!/usr/bin/env python3
"""
Initialize Yolo and process outputs
"""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo instance.

        Parameters:
        - model_path: path to where a Darknet Keras model is stored
        - classes_path: path to where the list of class names used for
            the Darknet model is found
        - class_t: float representing the box score threshold for the
            initial filtering step
        - nms_t: float representing the IOU threshold for non-max
            suppression
        - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = file.read().strip().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Applies sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs of the Darknet model

        Parameters:
        - outputs: list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image
        - image_size: numpy.ndarray containing the image’s original
            size [image_height, image_width]

        Returns:
        - boxes: list of numpy.ndarrays containing the processed
            boundary boxes for each output
        - box_confidences: list of numpy.ndarrays containing the box
            confidences for each output
        - box_class_probs: list of numpy.ndarrays containing the box’s
            class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            # Extract the components
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]
            tc = output[..., 4]
            tclasses = output[..., 5:]

            # Apply sigmoid to tx, ty, tc, tclasses
            bx = self.sigmoid(tx)
            by = self.sigmoid(ty)
            box_confidence = self.sigmoid(tc)[..., np.newaxis]
            class_probs = self.sigmoid(tclasses)

            # Create grid of coordinates
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid = cx_grid[..., np.newaxis]
            cy_grid = cy_grid[..., np.newaxis]

            bx += cx_grid
            by += cy_grid
            bx /= grid_w
            by /= grid_h

            bw = np.exp(tw) * anchors[:, 0]
            bh = np.exp(th) * anchors[:, 1]
            bw /= input_w
            bh /= input_h

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter the bounding boxes using the threshold.

        Parameters:
        - boxes: list of numpy.ndarrays containing the processed boundary
            boxes for each output
        - box_confidences: list of numpy.ndarrays containing the processed
            box confidences for each output
        - box_class_probs: list of numpy.ndarrays containing the processed
            box class probabilities for each output

        Returns:
        - filtered_boxes: numpy.ndarray containing all of the filtered
            bounding boxes
        - box_classes: numpy.ndarray containing the class number that
            each box in filtered_boxes predicts
        - box_scores: numpy.ndarray containing the box scores for each
            box in filtered_boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_conf, box_class_prob in zip(boxes, box_confidences,
                                                 box_class_probs):
            # Compute box scores
            scores = box_conf * box_class_prob
            max_scores = np.max(scores, axis=-1)
            max_classes = np.argmax(scores, axis=-1)

            # Filter boxes by score threshold
            mask = max_scores >= self.class_t
            filtered_boxes.append(box[mask])
            box_classes.append(max_classes[mask])
            box_scores.append(max_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores
