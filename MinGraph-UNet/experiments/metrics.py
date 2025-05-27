# experiments/metrics.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix # For precision, recall, f1, iou from CM

def segmentation_metrics(true_masks_flat, pred_masks_flat, num_classes, smooth=1e-6):
    """
    Computes segmentation metrics: IoU, Precision, Recall, F1 per class and averaged.
    Assumes masks are flattened: (N_pixels,) where N_pixels = B*H*W.
    true_masks_flat: Ground truth labels.
    pred_masks_flat: Predicted labels.
    """
    # Ensure inputs are numpy arrays
    if isinstance(true_masks_flat, torch.Tensor):
        true_masks_flat = true_masks_flat.cpu().numpy()
    if isinstance(pred_masks_flat, torch.Tensor):
        pred_masks_flat = pred_masks_flat.cpu().numpy()

    # Create a confusion matrix for all pixels
    # Labels argument ensures all classes are represented, even if not present in this batch/image
    cm = confusion_matrix(true_masks_flat, pred_masks_flat, labels=list(range(num_classes)))
    
    # Per-class metrics
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for cls_idx in range(num_classes):
        tp = cm[cls_idx, cls_idx]
        fp = np.sum(cm[:, cls_idx]) - tp # Sum of column cls_idx - TP
        fn = np.sum(cm[cls_idx, :]) - tp # Sum of row cls_idx - TP
        # tn = np.sum(cm) - tp - fp - fn # Not needed for these metrics

        # IoU (Jaccard Index)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        iou_per_class.append(iou)

        # Precision
        precision = (tp + smooth) / (tp + fp + smooth)
        precision_per_class.append(precision)

        # Recall (Sensitivity)
        recall = (tp + smooth) / (tp + fn + smooth)
        recall_per_class.append(recall)

        # F1-Score
        f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)
        f1_per_class.append(f1)
        
    # Mean metrics (Macro average - unweighted mean of per-class metrics)
    mean_iou = np.nanmean(iou_per_class) # nanmean to handle classes not in GT or preds
    mean_precision = np.nanmean(precision_per_class)
    mean_recall = np.nanmean(recall_per_class)
    mean_f1 = np.nanmean(f1_per_class)
    
    # Store all results
    results = {
        'iou_per_class': iou_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'mean_iou': mean_iou,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'confusion_matrix': cm
    }
    return results

def object_detection_mAP(gt_boxes_list, pred_boxes_list, iou_threshold=0.5, num_classes=1):
    """
    Simplified mAP@iou_threshold calculation for object detection.
    This is a complex metric, a full implementation (e.g. COCO eval) is extensive.
    This version is illustrative for a single class or averaged over classes.

    Args:
        gt_boxes_list (list of list of dicts): 
            List of images. Each image is a list of GT boxes.
            Each box dict: {'bbox': [x_min, y_min, x_max, y_max], 'class_id': int, 'used': False}
        pred_boxes_list (list of list of dicts): 
            List of images. Each image is a list of predicted boxes.
            Each box dict: {'bbox': [x_min, y_min, x_max, y_max], 'class_id': int, 'confidence': float}
        iou_threshold (float): IoU threshold for a match to be a True Positive.
        num_classes (int): Number of object classes.
    
    Returns:
        float: mAP value.
    """
    # This is a placeholder for a proper mAP calculation.
    # A full mAP involves:
    # 1. For each class:
    #    a. Match predictions to ground truths based on IoU.
    #    b. Sort predictions by confidence.
    #    c. Compute Precision-Recall curve.
    #    d. Calculate Average Precision (AP) from the PR curve.
    # 2. Average AP over all classes to get mAP.
    # Libraries like pycocotools are often used for this.
    print("Warning: mAP calculation is a placeholder. For accurate mAP, use a library like pycocotools.")
    
    # Illustrative simplified AP for a single class (class_id=0)
    # Assume num_classes = 1 for this example
    
    tp_all = 0
    fp_all = 0
    num_gt_total = 0

    for img_idx in range(len(gt_boxes_list)):
        gt_boxes_img = [box.copy() for box in gt_boxes_list[img_idx]] # Copy to modify 'used' flag
        pred_boxes_img = sorted(pred_boxes_list[img_idx], key=lambda x: x['confidence'], reverse=True)
        
        num_gt_total += len(gt_boxes_img)

        for pred_box in pred_boxes_img:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes_img):
                if gt_box['class_id'] == pred_box['class_id'] and not gt_box['used']:
                    # Calculate IoU (helper function needed)
                    iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp_all += 1
                gt_boxes_img[best_gt_idx]['used'] = True
            else:
                fp_all += 1
    
    # Simplified Precision and Recall (not a PR curve)
    precision = tp_all / (tp_all + fp_all + smooth) if (tp_all + fp_all) > 0 else 0
    recall = tp_all / (num_gt_total + smooth) if num_gt_total > 0 else 0
    
    # This is NOT mAP, just an example of how TP/FP might be counted.
    # mAP requires the full PR curve integration.
    # For the paper's table, you likely used a standard evaluation script (e.g., COCO).
    # Let's return a dummy value.
    return (precision + recall) / 2 if (precision + recall) > 0 else 0 # Highly simplified "AP"

def calculate_iou(box1, box2):
    """ Calculate IoU of two bounding boxes [xmin, ymin, xmax, ymax] """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def yield_estimation_metrics(gt_counts, pred_counts, 
                             gt_objects_list=None, pred_objects_list=None, 
                             matching_iou_thresh=0.5):
    """
    Computes yield estimation metrics.
    Args:
        gt_counts (list of int): List of ground truth object counts per image.
        pred_counts (list of int): List of predicted object counts per image.
        gt_objects_list (list of list of dicts, optional): For Object Matching Rate & Occlusion Robustness.
            Each dict: {'bbox': [x,y,w,h], 'class_id': int, 'occluded': bool (optional)}
        pred_objects_list (list of list of dicts, optional): For Object Matching Rate & Occlusion Robustness.
            Each dict: {'bbox': [x,y,w,h], 'class_id': int, 'confidence': float}
        matching_iou_thresh (float): IoU threshold for matching objects.

    Returns:
        dict: Dictionary of yield metrics.
    """
    gt_counts = np.array(gt_counts)
    pred_counts = np.array(pred_counts)

    # Count Accuracy (%)
    # This can be interpreted in a few ways.
    # 1. Exact match accuracy: sum(gt_counts == pred_counts) / len(gt_counts)
    # 2. Or based on total counts: sum(min(gt,pred)) / sum(gt) relative to sum(gt)/sum(pred)
    # Paper table has "Count Accuracy (%)", likely meaning something like:
    # 100 * (1 - MeanAbsoluteError / Mean(GT_Counts)) OR overall count accuracy.
    # Let's use a simple relative error for now, or average absolute error.
    # Paper's 95.3% count accuracy suggests a high level of agreement.
    # Let's use total predicted vs total GT.
    count_accuracy = (1.0 - np.abs(np.sum(pred_counts) - np.sum(gt_counts)) / (np.sum(gt_counts) + smooth)) * 100
    
    # Yield Estimation Error (%)
    # Paper: "yield estimation error is reduced to 5.9%"
    # This is often Mean Absolute Percentage Error (MAPE) or similar relative error.
    # MAPE = mean( |(gt - pred) / gt| ) * 100
    # Handle gt_counts being zero to avoid division by zero.
    valid_indices_for_mape = gt_counts > 0
    if np.any(valid_indices_for_mape):
        mape = np.mean(np.abs((gt_counts[valid_indices_for_mape] - pred_counts[valid_indices_for_mape]) / gt_counts[valid_indices_for_mape])) * 100
    else:
        mape = 0 if np.sum(np.abs(gt_counts - pred_counts)) == 0 else float('inf') # Or handle as appropriate
    yield_error = mape


    # Object Matching Rate (%) and Occlusion Robustness (%)
    # These require bounding box level GT and predictions.
    obj_matching_rate = -1.0 # Placeholder
    occlusion_robustness = -1.0 # Placeholder

    if gt_objects_list and pred_objects_list:
        total_gt_objects = 0
        matched_gt_objects = 0
        total_occluded_gt = 0
        matched_occluded_gt = 0

        for i in range(len(gt_objects_list)):
            gt_img_objs = [obj.copy() for obj in gt_objects_list[i]] # Mark 'used'
            for obj in gt_img_objs: obj['used'] = False
            pred_img_objs = sorted(pred_objects_list[i], key=lambda x: x.get('confidence', 1.0), reverse=True)

            total_gt_objects += len(gt_img_objs)
            if any(obj.get('occluded', False) for obj in gt_img_objs): # Simple check if any occluded
                # A more precise way: sum of occluded flags.
                total_occluded_gt += sum(1 for obj in gt_img_objs if obj.get('occluded',False))


            for pred_obj in pred_img_objs:
                best_iou = 0
                best_gt_match_idx = -1
                for gt_idx, gt_obj in enumerate(gt_img_objs):
                    if not gt_obj['used'] and gt_obj['class_id'] == pred_obj['class_id']:
                        iou = calculate_iou(pred_obj['bbox'], gt_obj['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_match_idx = gt_idx
                
                if best_iou >= matching_iou_thresh and best_gt_match_idx != -1:
                    gt_img_objs[best_gt_match_idx]['used'] = True
                    matched_gt_objects +=1
                    if gt_img_objs[best_gt_match_idx].get('occluded', False):
                        matched_occluded_gt += 1
        
        obj_matching_rate = (matched_gt_objects / (total_gt_objects + smooth)) * 100
        occlusion_robustness = (matched_occluded_gt / (total_occluded_gt + smooth)) * 100 if total_occluded_gt > 0 else -1.0 # Or 100 if no occluded GT

    results = {
        'count_accuracy_perc': count_accuracy,
        'yield_estimation_error_perc': yield_error,
        'object_matching_rate_perc': obj_matching_rate,
        'occlusion_robustness_perc': occlusion_robustness,
        'total_gt_count_sum': np.sum(gt_counts),
        'total_pred_count_sum': np.sum(pred_counts)
    }
    return results

if __name__ == '__main__':
    # Test Segmentation Metrics
    print("--- Testing Segmentation Metrics ---")
    true_m = torch.tensor([0, 1, 0, 1, 1, 0])
    pred_m = torch.tensor([0, 1, 1, 1, 0, 0])
    num_c = 2
    seg_res = segmentation_metrics(true_m, pred_m, num_c)
    print(f"IoU per class: {seg_res['iou_per_class']}")
    print(f"Mean IoU: {seg_res['mean_iou']:.4f}")
    print(f"Mean Precision: {seg_res['mean_precision']:.4f}")
    print(f"Mean Recall: {seg_res['mean_recall']:.4f}")
    print(f"Mean F1: {seg_res['mean_f1']:.4f}")

    # Test Yield Estimation Metrics
    print("\n--- Testing Yield Estimation Metrics ---")
    gt_c = [10, 12, 8, 15]
    pred_c = [9, 13, 7, 14]
    yield_res = yield_estimation_metrics(gt_c, pred_c)
    print(f"Count Accuracy: {yield_res['count_accuracy_perc']:.2f}%")
    print(f"Yield Estimation Error (MAPE): {yield_res['yield_estimation_error_perc']:.2f}%")
    
    # Test with bbox data
    gt_obj_list = [
        [{'bbox':[10,10,50,50],'class_id':0,'occluded':False}, {'bbox':[60,60,100,100],'class_id':0,'occluded':True}], # Img 1
        [{'bbox':[20,20,70,70],'class_id':0,'occluded':False}] # Img 2
    ]
    pred_obj_list = [
        [{'bbox':[12,12,48,48],'class_id':0,'confidence':0.9}, {'bbox':[50,50,90,90],'class_id':0,'confidence':0.8}], # Img 1 preds
        [{'bbox':[25,25,75,75],'class_id':0,'confidence':0.95}, {'bbox':[100,100,120,120],'class_id':0,'confidence':0.7}] # Img 2 preds
    ]
    yield_res_bbox = yield_estimation_metrics(gt_c=[2,1], pred_c=[2,2], # Counts based on above lists
                                        gt_objects_list=gt_obj_list, 
                                        pred_objects_list=pred_obj_list)
    print(f"Object Matching Rate: {yield_res_bbox['object_matching_rate_perc']:.2f}%")
    print(f"Occlusion Robustness: {yield_res_bbox['occlusion_robustness_perc']:.2f}%")