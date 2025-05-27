import torch
import yaml
import os
import argparse
import cv2
import numpy as np
import torch.nn.functional as F

# Project structure imports
from model.unet.unet_model import UNet # Assuming we infer with U-Net for segmentation
# If inferring with the full MinGraph-UNet up to a segmentation map:
# from ... (would need to instantiate the full pipeline)

from preprocessing.image_preprocessing.image_preprocess import ImagePreprocessor

def load_config(config_dir, config_name):
    with open(os.path.join(config_dir, config_name), 'r') as f:
        return yaml.safe_load(f)

def postprocess_segmentation(seg_logits_or_probs, num_classes):
    """ Converts model output to a displayable segmentation map. """
    if seg_logits_or_probs.ndim == 4: # B, C, H, W
        seg_logits_or_probs = seg_logits_or_probs.squeeze(0) # C, H, W (for single image inference)

    if seg_logits_or_probs.shape[0] == num_classes: # Logits or probabilities
        pred_labels = torch.argmax(seg_logits_or_probs, dim=0) # H, W
    else: # Already class labels (e.g. from a step that outputs hard labels)
        pred_labels = seg_logits_or_probs.long()

    # Convert to a color map for visualization
    # Example: simple color mapping
    h, w = pred_labels.shape
    output_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Define colors for classes (BGR for OpenCV)
    # Color for class 0 (background), class 1 (mango), etc.
    colors = [
        (0, 0, 0),       # Background - Black
        (0, 255, 0),     # Class 1 (e.g., Mango) - Green
        (0, 0, 255),     # Class 2 - Red
        (255, 0, 0),     # Class 3 - Blue
    ]
    # Extend colors if num_classes > len(colors)
    for i in range(len(colors), num_classes +1): # +1 for safety if classes are 1-indexed in some context
        colors.append(np.random.randint(0,255,3).tolist())


    for class_idx in range(num_classes):
        output_vis[pred_labels.cpu().numpy() == class_idx] = colors[class_idx]
        
    return pred_labels.cpu().numpy(), output_vis


def infer_segmentation(config_path, image_path, model_weights_path, output_dir="outputs/inference_results"):
    # --- Configuration Loading ---
    dataset_cfg = load_config(config_path, "dataset.yaml")
    model_cfg = load_config(config_path, "model.yaml")
    preproc_cfg = load_config(config_path, "preprocessing.yaml")
    # train_cfg might have device info if not hardcoded
    train_cfg = load_config(config_path, "training.yaml")


    device = torch.device(train_cfg.get('device', 'cpu') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Initialize Preprocessor ---
    image_preprocessor = ImagePreprocessor(
        resize_dim=tuple(preproc_cfg['resize_dim']),
        mean=tuple(preproc_cfg['normalization_mean']),
        std=tuple(preproc_cfg['normalization_std']),
        apply_augmentation=False # No augmentation for inference
    )

    # --- Load Model (Assuming U-Net for segmentation inference) ---
    # If full MinGraph-UNet is used for segmentation, instantiate it here.
    # For simplicity, this script uses the standalone U-Net.
    unet_config = model_cfg['unet']
    model = UNet(
        in_channels=unet_config['in_channels'],
        num_classes=unet_config['out_channels'],
        init_features=unet_config['init_features'],
        depth=unet_config['depth']
    ).to(device)

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
    
    checkpoint = torch.load(model_weights_path, map_location=device)
    # Check if weights are directly model.state_dict() or nested in a checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded from {model_weights_path}")

    # --- Preprocess Input Image ---
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")
    
    input_image_tensor = image_preprocessor.preprocess(image_path).unsqueeze(0).to(device) # Add batch dim

    # --- Perform Inference ---
    with torch.no_grad():
        # If using U-Net directly:
        seg_logits, _, _ = model(input_image_tensor)
        # If using full MinGraph-UNet, the forward pass would be more complex,
        # and you'd need to decide which output represents the final segmentation.
        # E.g., from fused_features passed through a final segmentation layer.
        
    # --- Postprocess and Save ---
    num_output_classes = unet_config['out_channels']
    predicted_labels_np, visualization_np = postprocess_segmentation(seg_logits.cpu(), num_output_classes)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_label_path = os.path.join(output_dir, f"{base_filename}_seg_labels.png")
    output_vis_path = os.path.join(output_dir, f"{base_filename}_seg_visualization.png")

    # Save the raw label map (e.g., for evaluation)
    # Ensure labels are in a savable format (e.g., uint8)
    cv2.imwrite(output_label_path, predicted_labels_np.astype(np.uint8))
    print(f"Saved label map to {output_label_path}")

    # Save the visual segmentation map
    cv2.imwrite(output_vis_path, visualization_np)
    print(f"Saved visualization to {output_vis_path}")

    # Optionally display
    # cv2.imshow("Input Image", cv2.imread(image_path))
    # cv2.imshow("Segmentation Visualization", visualization_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform segmentation inference using a trained model.")
    parser.add_argument('--config_path', type=str, default="configs/", help="Path to config files directory.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to the trained model weights (.pth file).")
    parser.add_argument('--output_dir', type=str, default="outputs/inference_results/", help="Directory to save inference results.")
    args = parser.parse_args()

    # Create dummy configs if not present (similar to train_segmentation.py)
    if not os.path.exists(args.config_path): os.makedirs(args.config_path)
    if not os.path.exists(os.path.join(args.config_path, "dataset.yaml")):
        with open(os.path.join(args.config_path, "dataset.yaml"), 'w') as f: yaml.dump({'num_classes': 2},f) # Minimal
    if not os.path.exists(os.path.join(args.config_path, "model.yaml")):
        with open(os.path.join(args.config_path, "model.yaml"), 'w') as f:
            yaml.dump({'unet': {'in_channels': 3, 'out_channels': 2, 'init_features': 16, 'depth': 2}},f)
    if not os.path.exists(os.path.join(args.config_path, "preprocessing.yaml")):
        with open(os.path.join(args.config_path, "preprocessing.yaml"), 'w') as f:
            yaml.dump({'resize_dim': [128,128], 'normalization_mean': [0.485,0.456,0.406], 'normalization_std': [0.229,0.224,0.225]},f)
    if not os.path.exists(os.path.join(args.config_path, "training.yaml")):
         with open(os.path.join(args.config_path, "training.yaml"), 'w') as f: yaml.dump({'device': 'cpu'},f)


    # Create a dummy model weights file and a dummy image for testing
    dummy_weights_path = "dummy_unet_weights.pth"
    if not os.path.exists(dummy_weights_path):
        # Need to match the model config in dummy model.yaml
        temp_model = UNet(in_channels=3, num_classes=2, init_features=16, depth=2)
        torch.save(temp_model.state_dict(), dummy_weights_path)
        print(f"Created dummy model weights at {dummy_weights_path}")

    dummy_image_file_infer = "temp_infer_image.png"
    if not os.path.exists(dummy_image_file_infer) or args.image_path == "temp_infer_image.png": # If user specifies this name
        cv2.imwrite(dummy_image_file_infer, np.random.randint(0,255,(200,200,3),dtype=np.uint8))
        print(f"Created dummy image at {dummy_image_file_infer}")
        # If args.image_path was not this dummy name, the user's path will be used.
        # If it WAS this dummy name, we need to ensure it's used if no other image is given.
        if args.image_path == "temp_infer_image.png" or args.image_path is None : # Default to dummy if no specific image provided
             current_image_path = dummy_image_file_infer
        else:
            current_image_path = args.image_path

    # Determine which image and weights to use for the test run
    test_image_path = args.image_path if args.image_path != "temp_infer_image.png" and os.path.exists(args.image_path) else dummy_image_file_infer
    test_weights_path = args.model_weights if os.path.exists(args.model_weights) else dummy_weights_path


    print(f"Running inference test with image: {test_image_path} and weights: {test_weights_path}")
    infer_segmentation(args.config_path, test_image_path, test_weights_path, args.output_dir)

    # Clean up dummy files created by this test
    if os.path.exists(dummy_weights_path) and args.model_weights != dummy_weights_path : os.remove(dummy_weights_path)
    if os.path.exists(dummy_image_file_infer) and args.image_path != dummy_image_file_infer : os.remove(dummy_image_file_infer)
    # output dir is not removed by default