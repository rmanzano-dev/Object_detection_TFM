from collections import defaultdict

import numpy as np
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion

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
    return metrics

import torch
from torchmetrics import Metric
from torchvision.ops import box_iou

class PrecisionRecallCurveObjectDetection(Metric):
    """
    Custom Precision-Recall Curve for Object Detection, with IoU thresholding for positive matches.
    """
    def __init__(self, iou_threshold=0.5, num_classes=80, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        # Initialize tensors to store predictions and targets for calculating precision-recall
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        """
        Update state with predictions and targets.
        
        Parameters:
            preds (list of dicts): Each dict has `boxes`, `scores`, and `labels` keys for predicted bounding boxes.
            targets (list of dicts): Each dict has `boxes` and `labels` keys for ground truth bounding boxes.
        """
        for pred, target in zip(preds, targets):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]
            
            target_boxes = target["boxes"]
            target_labels = target["labels"]

            # Add predictions and ground truths to respective lists
            self.preds.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
            self.targets.append({"boxes": target_boxes, "labels": target_labels})

    def compute(self):
        """
        Compute the precision-recall curve.
        Returns:
            precisions (list of Tensors): Precision values per class across different confidence thresholds.
            recalls (list of Tensors): Recall values per class across different confidence thresholds.
            thresholds (list of Tensors): Confidence thresholds corresponding to precision-recall points.
        """
        all_precisions = []
        all_recalls = []
        all_thresholds = []

        # Process each class separately
        for cls in range(self.num_classes):
            cls_preds = []
            cls_targets = []

            # Collect predictions and targets for this class
            for pred, target in zip(self.preds, self.targets):
                # Filter predictions and targets by class
                cls_pred_mask = pred["labels"] == cls
                cls_target_mask = target["labels"] == cls
                
                pred_boxes_cls = pred["boxes"][cls_pred_mask]
                pred_scores_cls = pred["scores"][cls_pred_mask]
                target_boxes_cls = target["boxes"][cls_target_mask]

                # Add to class-specific lists
                cls_preds.append((pred_boxes_cls, pred_scores_cls))
                cls_targets.append(target_boxes_cls)

            # Concatenate all predictions and targets for this class
            all_pred_boxes = torch.cat([p[0] for p in cls_preds])
            all_scores = torch.cat([p[1] for p in cls_preds])
            all_target_boxes = torch.cat(cls_targets)

            # Sort predictions by confidence score (descending)
            sorted_scores, sorted_idx = all_scores.sort(descending=True)
            sorted_pred_boxes = all_pred_boxes[sorted_idx]

            # Calculate TP and FP with IoU thresholding
            tp = torch.zeros_like(sorted_scores)
            fp = torch.zeros_like(sorted_scores)
            if len(all_target_boxes) == 0:
                fp[:] = 1
            else:
                matched_gt = set()
                for i, pred_box in enumerate(sorted_pred_boxes):
                    ious = box_iou(pred_box.unsqueeze(0), all_target_boxes).squeeze(0)
                    max_iou, max_idx = ious.max(0)
                    if max_iou >= self.iou_threshold and max_idx.item() not in matched_gt:
                        tp[i] = 1
                        matched_gt.add(max_idx.item())
                    else:
                        fp[i] = 1

            # Calculate cumulative TP and FP for precision-recall
            tp_cumsum = tp.cumsum(0)
            fp_cumsum = fp.cumsum(0)

            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recalls = tp_cumsum / len(all_target_boxes)

            all_precisions.append(precisions)
            all_recalls.append(recalls)
            all_thresholds.append(sorted_scores)

        return all_precisions, all_recalls, all_thresholds
