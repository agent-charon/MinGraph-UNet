# experiments/yield_estimation_performance.py
import torch
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import numpy as np
from tqdm import tqdm

# Project imports
# For yield estimation, the model needs to output counts or detections.
# This might be the full MinGraph-UNet ending in the DetectionHead.
# Or a specific YOLO-based model, FruitNet, MVCounter, etc.
# from model.full_mingraph_unet import MinGraphUNet_WithDetection # You'd need such a wrapper
from model.unet.unet_model import UNet # As a base if other parts are added
from model.fusion_detection.detection_head import DetectionHead # If used standalone or part of a larger model

from preprocessing.image_preprocessing.image_preprocess import ImagePreprocessor
from utils.mango_dataset import MangoDataset # Assuming it can also provide detection GT if needed
from experiments.metrics import yield_estimation_metrics

def load_config(config_dir, config_name):
    with open(os.path.join(config_dir, config_name), 'r') as f:
        return yaml.safe_load(f)

def evaluate_yield_model(config_path, model_type, model_weights_path):
    # --- Configuration ---
    dataset_cfg = load_config(config_path, "dataset.yaml")
    model_cfg = load_config(config_path, "model.yaml")
    preproc_cfg = load_config(config_path, "preprocessing.yaml")
    eval_cfg = load_config(config_path, "training.yaml")

    device = torch.device(eval_cfg.get('device', 'cpu') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for {model_type} yield evaluation.")

    # --- Preprocessor and Dataset ---
    # Dataset needs to provide ground truth counts per image.
    # For Object Matching Rate / Occlusion Robustness, it needs GT bounding boxes & occlusion flags.
    # Modifying MangoDataset for this or using a different dataset class might be needed.
    preprocessor = ImagePreprocessor(
        resize_dim=tuple(preproc_cfg['resize_dim']),
        mean=tuple(preproc_cfg['normalization_mean']),
        std=tuple(preproc_cfg['normalization_std']),
        apply_augmentation=False
    )
    test_dataset_path = os.path.join(dataset_cfg['data_root'], dataset_cfg.get('test_dir', 'test/'))
    if not os.path.exists(os.path.join(test_dataset_path, dataset_cfg['image_folder'])):
         print(f"Warning: Test image folder not found. Using val for test.")
         test_dataset_path = os.path.join(dataset_cfg['data_root'], dataset_cfg.get('val_dir', 'val/'))


    # For yield, MangoDataset __getitem__ would need to return (image, gt_count, gt_bboxes_for_image)
    # This is a simplification; current MangoDataset returns (image, mask).
    # We'll assume for this script that the loader provides (images, gt_counts_batch, gt_bboxes_batch_list).
    print("Warning: MangoDataset used for yield needs to provide GT counts and bboxes.")
    # Let's create a dummy version of MangoDataset for yield here for testing.
    class DummyYieldDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, preprocessor):
            self.num_samples = num_samples
            self.preprocessor = preprocessor
            self.img_size = preprocessor.resize_dim # H, W
        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            dummy_img_arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            img_tensor = self.preprocessor.preprocess(dummy_img_arr)
            gt_count = np.random.randint(1, 5) # Random count
            gt_bboxes_this_image = []
            for _ in range(gt_count):
                x1 = np.random.randint(0, self.img_size[1]//2)
                y1 = np.random.randint(0, self.img_size[0]//2)
                w = np.random.randint(10, self.img_size[1]//2)
                h = np.random.randint(10, self.img_size[0]//2)
                gt_bboxes_this_image.append({
                    'bbox': [x1,y1,x1+w,y1+h], 'class_id':0, # Assuming single class for mango
                    'occluded': np.random.choice([True, False])
                })
            return img_tensor, gt_count, gt_bboxes_this_image

    test_dataset = DummyYieldDataset(num_samples=20, preprocessor=preprocessor)
    # Custom collate_fn for lists of bboxes
    def yield_collate_fn(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        gt_counts = torch.tensor([item[1] for item in batch], dtype=torch.long)
        gt_bboxes_list_of_lists = [item[2] for item in batch] # List of lists of dicts
        return images, gt_counts, gt_bboxes_list_of_lists

    test_loader = DataLoader(
        test_dataset, batch_size=eval_cfg.get('eval_batch_size', 2), 
        shuffle=False, num_workers=eval_cfg.get('num_workers', 0), # num_workers > 0 might have issues with dummy dataset
        collate_fn=yield_collate_fn
    )
    
    # --- Load Model ---
    # This model should output detections (bboxes, scores, class_ids) or at least counts.
    # For "MinGraph-UNet" in Table 2, this implies the full pipeline ending in DetectionHead.
    if model_type.lower() == 'mingraph-unet':
        # This requires instantiating the full end-to-end model from train_end_to_end.py
        # and loading its composite state_dict. This is very complex.
        # For now, use a placeholder that simulates detection output.
        print("Warning: Full MinGraph-UNet for yield eval is complex. Using a placeholder mock model.")
        class MockDetector(nn.Module): # Mock that outputs random detections
            def __init__(self): super().__init__()
            def forward(self, x_batch):
                B = x_batch.size(0)
                pred_counts_batch = []
                pred_bboxes_batch_list_of_lists = []
                for _ in range(B):
                    num_preds = np.random.randint(0, 6)
                    pred_counts_batch.append(num_preds)
                    current_img_preds = []
                    for _ in range(num_preds):
                        x1,y1,w,h = np.random.rand(4) * preprocessor.resize_dim[1] # Scale to image size
                        current_img_preds.append({
                            'bbox': [x1,y1,min(x1+w, preprocessor.resize_dim[1]),min(y1+h,preprocessor.resize_dim[0])],
                            'class_id': 0, 'confidence': np.random.rand()
                        })
                    pred_bboxes_batch_list_of_lists.append(current_img_preds)
                return torch.tensor(pred_counts_batch), pred_bboxes_batch_list_of_lists
        model = MockDetector().to(device)
        # Actual loading:
        # model = FullMinGraphUNetWithDetection(...)
        # model.load_state_dict(...)
    # Add other models: YOLO, FruitNet, MVCounter, etc.
    else:
        raise ValueError(f"Model type '{model_type}' for yield eval not supported.")

    # No weights to load for MockDetector unless specified
    if model_type.lower() != 'mingraph-unet' or not isinstance(model, MockDetector): # Avoid loading for mock
        if not os.path.exists(model_weights_path):
             raise FileNotFoundError(f"Weights for {model_type} not found at {model_weights_path}")
        # checkpoint = torch.load(model_weights_path, map_location=device)
        # model.load_state_dict(...) # Adapt based on how weights are saved
        print(f"{model_type} model weights would be loaded from {model_weights_path}")

    model.eval()

    # --- Evaluation Loop ---
    all_gt_counts = []
    all_pred_counts = []
    all_gt_objects_list_of_lists = [] # For OMR, OR
    all_pred_objects_list_of_lists = [] # For OMR, OR

    print(f"Evaluating {model_type} for yield estimation...")
    with torch.no_grad():
        for images, gt_counts_b, gt_bboxes_b_lol in tqdm(test_loader, desc="Evaluating Yield"):
            images = images.to(device)
            
            # Model inference (should return counts and/or detected bboxes)
            # pred_counts_b, pred_bboxes_b_lol = model(images) # Example output signature
            if model_type.lower() == 'mingraph-unet' and isinstance(model, MockDetector):
                 pred_counts_b, pred_bboxes_b_lol = model(images) # Mock model output
            else:
                 # Actual model forward pass
                 # This depends HEAVILY on your specific model's output for detection/counting
                 # pred_bboxes, pred_confidence, pred_class_scores = model(images) # If model from e2e
                 # pred_counts_b = derive_counts_from_detections(pred_bboxes, pred_confidence, conf_thresh=0.5)
                 # pred_bboxes_b_lol = format_detections_for_metrics(pred_bboxes, pred_confidence, pred_class_scores)
                 raise NotImplementedError(f"Actual inference for {model_type} yield not implemented.")


            all_gt_counts.extend(gt_counts_b.cpu().tolist())
            all_pred_counts.extend(pred_counts_b.cpu().tolist() if isinstance(pred_counts_b, torch.Tensor) else pred_counts_b)
            
            all_gt_objects_list_of_lists.extend(gt_bboxes_b_lol)
            all_pred_objects_list_of_lists.extend(pred_bboxes_b_lol)

    # --- Calculate Metrics ---
    yield_metrics_results = yield_estimation_metrics(
        all_gt_counts, all_pred_counts,
        all_gt_objects_list_of_lists, all_pred_objects_list_of_lists
    )

    print(f"\n--- Yield Estimation Performance for {model_type} ---")
    print(f"Count Accuracy:            {yield_metrics_results['count_accuracy_perc']:.2f}% "
          f"(GT Sum: {yield_metrics_results['total_gt_count_sum']}, Pred Sum: {yield_metrics_results['total_pred_count_sum']})")
    print(f"Yield Estimation Error:    {yield_metrics_results['yield_estimation_error_perc']:.2f}%")
    print(f"Object Matching Rate:      {yield_metrics_results['object_matching_rate_perc']:.2f}%")
    print(f"Occlusion Robustness:      {yield_metrics_results['occlusion_robustness_perc']:.2f}%")
    
    return yield_metrics_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Yield Estimation Model.")
    parser.add_argument('--config_path', type=str, default="configs/", help="Path to config files dir.")
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['baseline_yield', 'improved_tracking', 'unet_dfs', 'mvcounter', 'mingraph-unet'], 
                        help="Type of yield model to evaluate.")
    parser.add_argument('--model_weights', type=str, help="Path to trained model weights (optional for mock).")
    args = parser.parse_args()

    # Setup dummy configs
    if not os.path.exists(args.config_path): os.makedirs(args.config_path)
    with open(os.path.join(args.config_path, "dataset.yaml"), 'w') as f:
        yaml.dump({'data_root': 'temp_mango_dataset_yield_eval', 'test_dir': 'test', 'image_folder':'images'}, f) # No mask folder needed for dummy
    with open(os.path.join(args.config_path, "model.yaml"), 'w') as f: yaml.dump({},f) # Model specific config might be complex
    with open(os.path.join(args.config_path, "preprocessing.yaml"), 'w') as f:
        yaml.dump({'resize_dim': [128,128], 'normalization_mean': [0.485,0.456,0.406], 'normalization_std': [0.229,0.224,0.225]},f)
    with open(os.path.join(args.config_path, "training.yaml"), 'w') as f:
         yaml.dump({'device':'cpu', 'eval_batch_size': 2, 'num_workers':0},f) # Use eval_batch_size

    # No dummy data creation here as DummyYieldDataset generates it internally for test.
    # No dummy weights needed if using MockDetector for mingraph-unet.

    try:
        # For non-mock models, model_weights would be required.
        weights = args.model_weights if args.model_weights else "dummy_yield_weights.pth"
        if args.model_type != "mingraph-unet" and not os.path.exists(weights): # Create dummy for others
            with open(weights, 'w') as wf: wf.write("dummy")
            print(f"Created dummy weights: {weights}")

        evaluate_yield_model(args.config_path, args.model_type, weights)
        
        if weights == "dummy_yield_weights.pth" and os.path.exists(weights):
             os.remove(weights)
    except Exception as e:
        print(f"ERROR during yield evaluation test: {e}")
        import traceback
        traceback.print_exc()