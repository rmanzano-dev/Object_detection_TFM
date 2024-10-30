from collections import defaultdict

import numpy as np
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

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

        tg_boxes = target[0]['boxes'].cpu().numpy()
        tg_labels = target[0]['labels'].cpu().numpy()

        # Keep track of matched ground truths for each class
        matched_tg = defaultdict(set)

        # Sort predictions by score (highest first)
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]

        # Loop over each predicted box and find the best matching ground truth
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_tg_idx = -1

            # Look for a matching ground truth box with the same class
            for tg_idx, tg_box in enumerate(tg_boxes):
                if tg_labels[tg_idx] == pred_label and tg_idx not in matched_tg[pred_label]:
                    iou = compute_iou(pred_box, tg_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_tg_idx = tg_idx

            # If a match is found, count it as a true positive, else it's a false positive
            if best_tg_idx >= 0:
                true_positives[pred_label] += 1
                matched_tg[pred_label].add(best_tg_idx)
            else:
                false_positives[pred_label] += 1

        # False negatives are the unmatched ground truth boxes
        for tg_label in tg_labels:
            false_negatives[tg_label] += len(tg_boxes) - len(matched_tg[tg_label])

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

        ap_per_class[class_id] = compute_ap(torch.tensor([precision]), torch.tensor([recall]))

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

def get_precision_recall(predictions, targets, iou_threshold=0.5):
    """ Calculate precision and recall based on predictions and targets """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, target in tqdm(zip(predictions, targets)):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()

        tg_boxes = target['boxes'].cpu().numpy()
        tg_labels = target['labels'].cpu().numpy()

        matched_tg = set()

        # Sort predictions by score (highest first)
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_tg_idx = -1

            for tg_idx, tg_box in enumerate(tg_boxes):
                if tg_labels[tg_idx] == pred_label and tg_idx not in matched_tg:
                    iou = compute_iou(pred_box, tg_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_tg_idx = tg_idx

            if best_tg_idx >= 0:
                true_positives += 1
                matched_tg.add(best_tg_idx)
            else:
                false_positives += 1

        # Remaining ground truth boxes are false negatives
        false_negatives += len(tg_boxes) - len(matched_tg)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall

import torch

def compute_ap(precision, recall):
    # Ensure precision and recall are sorted in descending order of recall
    sorted_indices = torch.argsort(recall, descending=True)
    precision = precision[sorted_indices]
    recall = recall[sorted_indices]

    # Insert a zero at the beginning of the recall and precision arrays
    recall = torch.cat([torch.tensor([0]), recall])
    precision = torch.cat([torch.tensor([0]), precision])

    # Compute AP using the trapezoidal rule
    ap = torch.trapz(precision, recall)
    return ap.item()


def get_metrics(preds, targets):
    metric = MeanAveragePrecision(class_metrics=True, extended_summary=True, backend="faster_coco_eval")
    metric.update(preds, targets)

    print("Getting metrics...")

    metrics = metric.compute()
    precision_tensor = metrics['precision']
    recall_tensor = metrics["recall"]

    return metrics, precision_tensor


def get_precision_recall_per_class(predictions, targets, iou_threshold=0.5):
    """ Calculate precision and recall per class based on predictions and targets """
    # Initialize dictionaries to keep counts for each class
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for pred, target in tqdm(zip(predictions, targets)):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()

        tg_boxes = target['boxes'].cpu().numpy()
        tg_labels = target['labels'].cpu().numpy()

        # Track matched target boxes per label
        matched_tg = defaultdict(set)

        # Sort predictions by score (highest first)
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_tg_idx = -1

            for tg_idx, tg_box in enumerate(tg_boxes):
                if tg_labels[tg_idx] == pred_label and tg_idx not in matched_tg[pred_label]:
                    iou = compute_iou(pred_box, tg_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_tg_idx = tg_idx

            if best_tg_idx >= 0:
                true_positives[pred_label] += 1
                matched_tg[pred_label].add(best_tg_idx)
            else:
                false_positives[pred_label] += 1

        # Count remaining unmatched ground truth boxes as false negatives per class
        for tg_idx, tg_label in enumerate(tg_labels):
            if tg_idx not in matched_tg[tg_label]:
                false_negatives[tg_label] += 1

    # Calculate precision and recall per class
    precision_per_class = {}
    recall_per_class = {}
    for label in set(true_positives.keys()).union(false_positives.keys(), false_negatives.keys()):
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_per_class[label] = precision
        recall_per_class[label] = recall

    return precision_per_class, recall_per_class

def plot_precision_recall_curve(precision_tensor, num_classes):
    recall_values = np.linspace(0, 1, 101)
    plt.figure(figsize=(12, 6))

    classes = ["VEHÍCULO", "PEATÓN", "CICLISTA"]

    precision_values = []
    # Iterate through each class and the total
    for i, class_name in enumerate(classes):
        precision_values.append(precision_tensor[0,:,i,0,2]) # Shape (T, R, K, A, M)
        plt.plot(recall_values, precision_values[i], label=f'{class_name}', linewidth=2)

    avg_precision = torch.mean(precision_tensor, dim=2)[0,:, 0, 2]

    avg_prec_recall = torch.mean(avg_precision)
    plt.plot(recall_values, avg_precision, label=f'all classes {np.round(avg_prec_recall, decimals=3)} mAP@50', linewidth=5, linestyle="--")
    # Plotting the Precision-Recall curve
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall Curve para Waymo FasterR-CNN, desde cero')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    #plt.savefig("PR_Curve_Waymo.png")
    #plt.show()