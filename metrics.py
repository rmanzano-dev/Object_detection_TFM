from collections import defaultdict

import numpy as np
from tqdm import tqdm


def calculate_map_per_class(predictions, targets, num_classes, iou_threshold=0.5):
    # Calculate mAP@50 per class
    
    # Initialize dictionaries to store true positives, false positives, and false negatives per class
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    # Loop over each prediction-target pair
    for pred, target in zip(predictions, targets):
        pred_boxes = pred[0]['boxes'].cpu().numpy()
        pred_scores = pred[0]['scores'].cpu().numpy()
        pred_labels = pred[0]['labels'].cpu().numpy()

        gt_boxes = target[0]['boxes'].cpu().numpy()
        gt_labels = target[0]['labels'].cpu().numpy()

        # Keep track of matched ground truths for each class
        matched_gt = defaultdict(set)

        # Sort predictions by score (highest first)
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]

        # Loop over each predicted box and find the best matching ground truth
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1

            # Look for a matching ground truth box with the same class
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_labels[gt_idx] == pred_label and gt_idx not in matched_gt[pred_label]:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx

            # If a match is found, count it as a true positive, else it's a false positive
            if best_gt_idx >= 0:
                true_positives[pred_label] += 1
                matched_gt[pred_label].add(best_gt_idx)
            else:
                false_positives[pred_label] += 1

        # False negatives are the unmatched ground truth boxes
        for gt_label in gt_labels:
            false_negatives[gt_label] += len(gt_boxes) - len(matched_gt[gt_label])

    # Compute precision, recall, and average precision for each class
    precision_per_class = {}
    recall_per_class = {}
    ap_per_class = {}

    for class_id in range(1, num_classes):  # Assuming classes are labeled 1 to num_classes
        tp = true_positives[class_id]
        fp = false_positives[class_id]
        fn = false_negatives[class_id]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_per_class[class_id] = precision
        recall_per_class[class_id] = recall

        # For AP, if we only compute for a fixed IoU threshold (50%), AP = precision at recall = 1.
        # For a more complex AP calculation (e.g., for mAP), you would calculate precision at multiple recall thresholds.
        ap_per_class[class_id] = precision  # Simplification: AP = Precision at recall 1 for mAP50

    return precision_per_class, recall_per_class, ap_per_class

def compute_iou(box1, box2):
    # Compute IoU between two bounding boxes
    # Unpack the boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Compute intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # Compute areas of both boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute union
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_map50(predictions, targets, iou_threshold=0.5):
    """ Calculate mAP@50 based on predictions and targets """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, target in tqdm(zip(predictions, targets)):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()

        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        matched_gt = set()

        # Sort predictions by score (highest first)
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_labels[gt_idx] == pred_label and gt_idx not in matched_gt:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1

        # Remaining ground truth boxes are false negatives
        false_negatives += len(gt_boxes) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall