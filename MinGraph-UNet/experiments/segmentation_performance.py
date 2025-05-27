# experiments/segmentation_performance.py
import torch
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import numpy as np
from tqdm import tqdm

# Project imports
from model.unet.unet_model import UNet # Example for U-Net baseline
# from model.deeplabv3plus import DeepLabV3Plus # If you have this model
# from ... (Load your MinGraphUNet model - this is complex)
from preprocessing.image_preprocessing.image_preprocess import ImagePreprocessor
from utils.mango_dataset import MangoDataset
from experiments.metrics import segmentation_metrics

def load_config(config_dir, config_name):
    with open(os.path.join(config_dir, config_name), 'r') as f:
        return yaml.safe_load(f)

def evaluate_segmentation_model(config_path, model_type, model_weights_path):
    # --- Configuration ---
    dataset_cfg = load_config(config_path, "dataset.yaml")
    model_cfg = load_config(config_path, "model.yaml") # Specific to the model_type
    preproc_cfg = load_config(config_path, "preprocessing.yaml")
    eval_cfg = load_config(config_path, "training.yaml") # For device, batch_size

    device = torch.device(eval_cfg.get('device', 'cpu') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for {model_type} evaluation.")

    # --- Preprocessor and Dataset ---
    preprocessor = ImagePreprocessor(
        resize_dim=tuple(preproc_cfg['resize_dim']),
        mean=tuple(preproc_cfg['normalization_mean']),
        std=tuple(preproc_cfg['normalization_std']),
        apply_augmentation=False
    )
    test_dataset_path = os.path.join(dataset_cfg['data_root'], dataset_cfg.get('test_dir', 'test/')) # Use 'test_dir' or default
    if not os.path.exists(os.path.join(test_dataset_path, dataset_cfg['image_folder'])):
        print(f"Warning: Test image folder not found at {os.path.join(test_dataset_path, dataset_cfg['image_folder'])}. Using val for test.")
        test_dataset_path = os.path.join(dataset_cfg['data_root'], dataset_cfg.get('val_dir', 'val/'))


    test_dataset = MangoDataset(
        image_dir=os.path.join(test_dataset_path, dataset_cfg['image_folder']),
        mask_dir=os.path.join(test_dataset_path, dataset_cfg['mask_folder']),
        preprocessor=preprocessor,
        num_classes=get_config_recursively(model_cfg, f'{model_type.lower()}_config.unet.out_channels', # Flexible model config access
                                          default=get_config_recursively(model_cfg, 'unet.out_channels', 2)) # Default if model specific not found
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_cfg.get('eval_batch_size', eval_cfg['batch_size']), 
        shuffle=False, num_workers=eval_cfg.get('num_workers', 2)
    )

    # --- Load Model ---
    # This part needs to be flexible to load different model types
    num_output_classes = test_dataset.num_classes
    if model_type.lower() == 'unet':
        unet_config = get_config_recursively(model_cfg, 'unet')
        model = UNet(
            in_channels=unet_config['in_channels'], num_classes=num_output_classes,
            init_features=unet_config['init_features'], depth=unet_config['depth']
        ).to(device)
    elif model_type.lower() == 'mingraph-unet':
        # Loading the full MinGraph-UNet is complex.
        # It requires instantiating all its components and then loading state dicts for each.
        # For this script, we might assume its segmentation output comes from a specific part
        # or it's saved as an end-to-end model that directly outputs segmentation logits.
        # Placeholder:
        print("Warning: Full MinGraph-UNet loading for segmentation eval is complex and placeholder here.")
        # For now, let's assume MinGraph-UNet for segmentation uses a U-Net compatible structure output
        unet_config = get_config_recursively(model_cfg, 'unet') # Uses base U-Net config
        model = UNet(
            in_channels=unet_config['in_channels'], num_classes=num_output_classes,
            init_features=unet_config['init_features'], depth=unet_config['depth']
        ).to(device) # This is a simplification for loading
    # Add other model types like DeepLabV3+, YOLO-based segmentation, etc.
    else:
        raise ValueError(f"Model type '{model_type}' not supported for evaluation in this script.")

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights for {model_type} not found at {model_weights_path}")
    
    checkpoint = torch.load(model_weights_path, map_location=device)
    if 'model_state_dict' in checkpoint: # Common for own training scripts
        # If loading full MinGraph-UNet, state_dict might be for multiple sub-models
        if model_type.lower() == 'mingraph-unet':
             # Need to load parts: model.unet_model.load_state_dict(checkpoint['unet_model_state_dict']), etc.
             # This requires saving checkpoints appropriately in train_end_to_end.py
             try: # Try loading as a simple model first
                model.load_state_dict(checkpoint['model_state_dict'])
             except: # If that fails, it might be a composite state dict
                print(f"Complex state dict for MinGraph-UNet. Attempting to load U-Net part if key 'unet_model_state_dict' exists.")
                if 'unet_model_state_dict' in checkpoint: # Check if specific key for U-Net part exists
                    model.load_state_dict(checkpoint['unet_model_state_dict']) # Assuming 'model' here is just the U-Net part for eval
                elif 'unet' in checkpoint : # if keys are like unet.encoder...
                    # model.load_state_dict(checkpoint['unet']) # This might work if saved that way
                    # Filter keys for the U-Net part if MinGraph-UNet saved all params with prefixes
                    unet_state_dict = {k.replace('unet_model.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('unet_model.')}
                    if unet_state_dict: model.load_state_dict(unet_state_dict)
                    else: model.load_state_dict(checkpoint['model_state_dict']) # Fallback

                else: raise RuntimeError("Cannot load MinGraph-UNet state_dict for segmentation evaluation.")

        else: # For simple models like U-Net baseline
            model.load_state_dict(checkpoint['model_state_dict'])
    else: # Direct state_dict save
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"{model_type} model loaded from {model_weights_path}")

    # --- Evaluation Loop ---
    all_true_masks_flat = []
    all_pred_masks_flat = []

    print(f"Evaluating {model_type}...")
    with torch.no_grad():
        for images, true_masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            # true_masks are already (B,H,W) long tensors

            # Model inference (adapt based on model_type's output)
            if model_type.lower() == 'unet':
                seg_logits, _, _ = model(images)
            elif model_type.lower() == 'mingraph-unet':
                # This depends on how MinGraph-UNet's final segmentation is produced.
                # If it's from the initial U-Net logits:
                seg_logits, _, _ = model(images) # model here is the U-Net part
                # Or if F_fused is passed to a final segmentation conv layer.
                # This requires running the full pipeline, which is not set up here.
                # For Table 1, "Proposed Method" uses the full pipeline.
                # This script would need to run that full forward pass.
                # For now, this will evaluate the U-Net component if 'model' is just U-Net.
            else: # Add other models
                seg_logits = model(images) # Assuming standard output

            # Convert logits to predicted labels
            # seg_probs = torch.softmax(seg_logits, dim=1)
            pred_labels = torch.argmax(seg_logits, dim=1) # (B, H, W)

            all_true_masks_flat.append(true_masks.view(-1).cpu())
            all_pred_masks_flat.append(pred_labels.view(-1).cpu())

    # Concatenate all results
    final_true_flat = torch.cat(all_true_masks_flat)
    final_pred_flat = torch.cat(all_pred_masks_flat)

    # --- Calculate Metrics ---
    metrics_results = segmentation_metrics(final_true_flat, final_pred_flat, num_classes=num_output_classes)

    print(f"\n--- Segmentation Performance for {model_type} ---")
    # Assuming class 1 is 'mango' and class 0 is 'background'
    mango_class_idx = 1 
    if num_output_classes > mango_class_idx :
        print(f"Precision (Mango - Class {mango_class_idx}): {metrics_results['precision_per_class'][mango_class_idx]*100:.2f}%")
        print(f"Recall (Mango - Class {mango_class_idx}):    {metrics_results['recall_per_class'][mango_class_idx]*100:.2f}%")
        print(f"F1-Score (Mango - Class {mango_class_idx}):  {metrics_results['f1_per_class'][mango_class_idx]*100:.2f}%")
        print(f"IoU (Mango - Class {mango_class_idx}):       {metrics_results['iou_per_class'][mango_class_idx]*100:.2f}%")
    else: # Handle binary case where metrics might be for the positive class directly
        print(f"Precision (Foreground): {metrics_results['precision_per_class'][-1]*100:.2f}%") # Often last class is foreground
        print(f"Recall (Foreground):    {metrics_results['recall_per_class'][-1]*100:.2f}%")
        print(f"F1-Score (Foreground):  {metrics_results['f1_per_class'][-1]*100:.2f}%")
        print(f"IoU (Foreground):       {metrics_results['iou_per_class'][-1]*100:.2f}%")


    print(f"\nMean Precision (Macro Avg): {metrics_results['mean_precision']*100:.2f}%")
    print(f"Mean Recall (Macro Avg):    {metrics_results['mean_recall']*100:.2f}%")
    print(f"Mean F1-Score (Macro Avg):  {metrics_results['mean_f1']*100:.2f}%")
    print(f"Mean IoU (Macro Avg):       {metrics_results['mean_iou']*100:.2f}%")

    # mAP@50 is for object detection, not typically for semantic segmentation.
    # If your "YOLO-Based Approach" in Table 1 is a segmentation model (like YOLACT), then these metrics apply.
    # If it's an object detector whose boxes are converted to masks, then mAP applies to the detection part.
    # The paper table shows IoU, Precision, Recall for YOLO-based, so it seems segmentation-focused.

    return metrics_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Model Performance.")
    parser.add_argument('--config_path', type=str, default="configs/", help="Path to config files dir.")
    parser.add_argument('--model_type', type=str, required=True, choices=['unet', 'deeplabv3+', 'yolo_based', 'mingraph-unet'], help="Type of model to evaluate.")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to trained model weights.")
    args = parser.parse_args()

    # Create dummy data and configs for a quick test run
    if not os.path.exists(args.config_path): os.makedirs(args.config_path)
    # Simplified configs for testing the script structure
    with open(os.path.join(args.config_path, "dataset.yaml"), 'w') as f:
        yaml.dump({'data_root': 'temp_mango_dataset_eval', 'test_dir': 'test', 'val_dir':'val', 'image_folder': 'images', 'mask_folder': 'masks'}, f)
    with open(os.path.join(args.config_path, "model.yaml"), 'w') as f: # Generic, specific model params would be nested
        yaml.dump({'unet': {'in_channels': 3, 'out_channels': 2, 'init_features': 8, 'depth': 1}}, f)
    with open(os.path.join(args.config_path, "preprocessing.yaml"), 'w') as f:
        yaml.dump({'resize_dim': [64,64], 'normalization_mean': [0.485,0.456,0.406], 'normalization_std': [0.229,0.224,0.225]},f)
    with open(os.path.join(args.config_path, "training.yaml"), 'w') as f: # Re-using training for device/batch
         yaml.dump({'device':'cpu', 'batch_size': 2, 'eval_batch_size':2, 'num_workers':0},f)

    base_dir_test = "temp_mango_dataset_eval"
    test_img_dir = os.path.join(base_dir_test, "test", "images")
    test_mask_dir = os.path.join(base_dir_test, "test", "masks")
    if not os.path.exists(test_img_dir): os.makedirs(test_img_dir)
    if not os.path.exists(test_mask_dir): os.makedirs(test_mask_dir)
    for i in range(4):
        cv2.imwrite(os.path.join(test_img_dir, f"t_img_{i}.png"), np.random.randint(0,255,(100,100,3),dtype=np.uint8))
        cv2.imwrite(os.path.join(test_mask_dir, f"t_img_{i}.png"), np.random.randint(0,2,(100,100),dtype=np.uint8))

    dummy_weights_eval = f"dummy_{args.model_type}_weights.pth"
    if not os.path.exists(dummy_weights_eval) or args.model_weights == dummy_weights_eval:
        if args.model_type.lower() == 'unet' or args.model_type.lower() == 'mingraph-unet': # Assuming U-Net part for MinGraph
            temp_model_eval = UNet(in_channels=3, num_classes=2, init_features=8, depth=1)
            torch.save(temp_model_eval.state_dict(), dummy_weights_eval)
        else: # For other models, create a dummy file
            with open(dummy_weights_eval, 'w') as wf: wf.write("dummy weights")
        print(f"Created dummy weights at {dummy_weights_eval}")
    
    eval_weights_path = args.model_weights if os.path.exists(args.model_weights) else dummy_weights_eval

    try:
        evaluate_segmentation_model(args.config_path, args.model_type, eval_weights_path)
    except Exception as e:
        print(f"ERROR during segmentation evaluation test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        if os.path.exists(base_dir_test): shutil.rmtree(base_dir_test)
        if os.path.exists(dummy_weights_eval) and eval_weights_path == dummy_weights_eval : os.remove(dummy_weights_eval)