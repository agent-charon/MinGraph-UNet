import torch
import yaml
import os
import argparse
import numpy as np

# Add project root to Python path or use relative imports carefully
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.image_preprocessing.image_preprocess import ImagePreprocessor
from preprocessing.graph_construction.patch_graph_construction import PatchGraphConstructor
from preprocessing.graph_feature_processing.edge_detection import EdgeDetector
from preprocessing.graph_feature_processing.histogram_equalization import HistogramEqualizer
from preprocessing.graph_feature_processing.gaussian_smoothing import GaussianSmoother

from model.gat.graph_attention import GATNetwork
from model.graph_partition.mincut_refinement import MinCutRefinement # For L_partition
from model.unet.unet_encoder import UNetEncoder # To get initial patch features from U-Net

# For MinCutRefinement, we need a dummy segment predictor network for this script
import torch.nn as nn
class SimpleSegmentPredictor(nn.Module):
    def __init__(self, in_dim, out_segments):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_segments)
    def forward(self, x, edge_index=None): # edge_index might be used by GNN-based predictor
        return self.fc(x)


def load_config(config_dir, config_name):
    with open(os.path.join(config_dir, config_name), 'r') as f:
        return yaml.safe_load(f)

def test_graph_pipeline(config_path, dummy_image_path=None):
    # Load configurations
    dataset_cfg = load_config(config_path, "dataset.yaml")
    model_cfg = load_config(config_path, "model.yaml")
    preproc_cfg = load_config(config_path, "preprocessing.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Image Preprocessing
    image_preprocessor = ImagePreprocessor(
        resize_dim=tuple(preproc_cfg['resize_dim']),
        mean=tuple(preproc_cfg['normalization_mean']),
        std=tuple(preproc_cfg['normalization_std'])
    )
    if dummy_image_path is None: # Create a dummy image if not provided
        dummy_img_arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        img_tensor_normalized = image_preprocessor.preprocess(dummy_img_arr).unsqueeze(0).to(device) # Add batch dim
        img_rgb_resized_np = cv2.resize(cv2.cvtColor(dummy_img_arr, cv2.COLOR_BGR2RGB), 
                                     (preproc_cfg['resize_dim'][1], preproc_cfg['resize_dim'][0]))
    else:
        img_tensor_normalized = image_preprocessor.preprocess(dummy_image_path).unsqueeze(0).to(device)
        img_bgr_loaded = cv2.imread(dummy_image_path)
        img_rgb_resized_np = cv2.resize(cv2.cvtColor(img_bgr_loaded, cv2.COLOR_BGR2RGB),
                                     (preproc_cfg['resize_dim'][1], preproc_cfg['resize_dim'][0]))


    # 2. Initial Patch Features (from U-Net encoder + processed image features)
    #    a. U-Net encoder features (simplified for this script)
    unet_encoder_cfg = model_cfg['unet']
    # We need a U-Net encoder to get some base features for patches.
    # For simplicity, let's assume U-Net encoder outputs features that can be averaged per patch.
    # Or, directly use patch pixels + processed features.
    
    # Get Patches from the *original scale normalized image* before U-Net processes it for features
    patch_constructor = PatchGraphConstructor(patch_size=model_cfg['graph_construction']['patch_size'])
    # image_tensor_normalized is (B, C, H, W). Use first item in batch.
    patches_tensor, (nph, npw) = patch_constructor.image_to_patches(img_tensor_normalized.squeeze(0))
    num_patches = patches_tensor.shape[0]
    
    # For U-Net features per patch (placeholder): average patch pixels for now
    # A real implementation would pass patches through early U-Net layers or map U-Net features.
    unet_patch_features_dim = 16 # Dummy dimension
    unet_patch_features = patches_tensor.mean(dim=[1,2,3]).unsqueeze(-1).repeat(1, unet_patch_features_dim).to(device)
    print(f"Placeholder U-Net patch features shape: {unet_patch_features.shape}")

    #    b. Processed image features (Sobel, HistEq, Smooth) per patch
    #       These operate on the RGB image (0-255 range, H, W, C)
    edge_detector = EdgeDetector(kernel_size=preproc_cfg['sobel_kernel_size'])
    hist_equalizer = HistogramEqualizer()
    smoother = GaussianSmoother(kernel_size=tuple(preproc_cfg['gaussian_blur_kernel_size']), 
                                sigma_x=preproc_cfg['gaussian_blur_sigma'])

    # Apply to the resized RGB image (before normalization for ToTensor)
    sobel_map_full = edge_detector.sobel_edges(img_rgb_resized_np) # (H,W)
    histeq_img_full = hist_equalizer.equalize_histogram_rgb(img_rgb_resized_np) # (H,W,C)
    # Gaussian smoothing is usually on original or slightly processed, not on histeq
    # Let's smooth the original resized RGB for demonstration
    # smoothed_img_full = smoother.smooth(img_rgb_resized_np) # (H,W,C)

    # Convert these full processed images to patches and then average/flatten features
    # Sobel patches
    sobel_patches, _ = patch_constructor.image_to_patches(torch.from_numpy(sobel_map_full).float().unsqueeze(0).to(device)) # (N_p, 1, P_h, P_w)
    sobel_patch_features = sobel_patches.mean(dim=[1,2,3]).unsqueeze(-1).to(device) # (N_p, 1)
    
    # HistEq patches
    histeq_img_tensor = torch.from_numpy(histeq_img_full).float().permute(2,0,1).to(device) # (C,H,W)
    histeq_patches, _ = patch_constructor.image_to_patches(histeq_img_tensor) # (N_p, C, P_h, P_w)
    histeq_patch_features = histeq_patches.mean(dim=[1,2,3]).to(device) # (N_p, C)
    
    # Concatenate all patch features: U-Net_placeholder + Sobel + HistEq
    # Ensure dimensions are consistent for concatenation (e.g., all are flat vectors per patch)
    combined_patch_features = torch.cat([
        unet_patch_features, 
        sobel_patch_features, 
        histeq_patch_features
    ], dim=1).to(device)
    current_feature_dim = combined_patch_features.shape[1]
    print(f"Combined initial patch features shape: {combined_patch_features.shape}")

    # 3. Construct Patch Graph
    _, patch_edge_index = patch_constructor.construct_patch_graph(img_tensor_normalized.squeeze(0), combined_patch_features)
    patch_edge_index = patch_edge_index.to(device)
    print(f"Patch graph: {num_patches} nodes, {patch_edge_index.shape[1]} edges.")

    # 4. GAT (Patch-level refinement)
    gat_cfg = model_cfg['gat']
    patch_gat = GATNetwork(
        node_feature_dim=current_feature_dim,
        hidden_dim=gat_cfg['hidden_dim'],
        output_dim=gat_cfg['output_dim'], # This will be the input dim for segment predictor
        num_heads=gat_cfg['num_heads'],
        dropout_rate=gat_cfg['dropout'],
        alpha=gat_cfg['alpha'],
        num_gat_layers=1 # Example: 1 GAT layer for patches
    ).to(device)
    
    gat_refined_patch_feats = patch_gat(combined_patch_features, patch_edge_index)
    print(f"GAT-refined patch features shape: {gat_refined_patch_feats.shape}") # (N_patches, gat_output_dim)

    # 5. MinCut Refinement (L_partition)
    mincut_module_cfg = model_cfg.get('mincut', {}) # Get mincut config or empty dict
    num_segments_for_ncut = dataset_cfg.get('num_semantic_regions', 3) # e.g. mango, leaf, background
    
    # Predictor network for segment assignments (input dim is GAT output dim)
    segment_predictor = SimpleSegmentPredictor(gat_cfg['output_dim'], num_segments_for_ncut).to(device)
    
    mincut_refiner = MinCutRefinement(
        gamma_unet_priors=mincut_module_cfg.get('gamma_unet_priors', 0.5),
        # sigmas are for E(S) edge weights, not directly Ncut loss weights here
    )
    
    l_partition, s_star_soft_patches = mincut_refiner(
        gat_refined_patch_feats, 
        patch_edge_index, 
        num_segments_for_ncut, 
        segment_predictor
    )
    print(f"L_partition (Ncut loss): {l_partition.item()}")
    print(f"S*_soft_patches (patch assignments) shape: {s_star_soft_patches.shape}")

    hard_patch_labels = torch.argmax(s_star_soft_patches, dim=1)
    print(f"Example hard patch labels: {hard_patch_labels[:10]}")
    
    print("\nGraph refinement pipeline test completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Graph Refinement Pipeline.")
    parser.add_argument('--config_path', type=str, default="configs/", help="Path to config files dir.")
    parser.add_argument('--image', type=str, default=None, help="Optional path to a test image.")
    args = parser.parse_args()

    # Create dummy config files for testing if they don't exist (similar to train_segmentation.py)
    # Ensure all necessary keys are present in model.yaml (gat, graph_construction, mincut)
    # and preprocessing.yaml (sobel_kernel_size, etc.)

    if not os.path.exists(args.config_path): os.makedirs(args.config_path)
    # dataset.yaml (add num_semantic_regions)
    if not os.path.exists(os.path.join(args.config_path, "dataset.yaml")):
        with open(os.path.join(args.config_path, "dataset.yaml"), 'w') as f: yaml.dump({'num_semantic_regions': 3}, f)
    # model.yaml (add graph_construction, gat, mincut)
    if not os.path.exists(os.path.join(args.config_path, "model.yaml")):
        with open(os.path.join(args.config_path, "model.yaml"), 'w') as f:
            yaml.dump({
                'unet': {'init_features': 16, 'depth': 2},
                'graph_construction': {'patch_size': 16},
                'gat': {'hidden_dim': 32, 'output_dim': 16, 'num_heads': 2, 'dropout': 0.1, 'alpha': 0.2},
                'mincut': {'gamma_unet_priors': 0.5}
            }, f)
    # preprocessing.yaml (add sobel, gaussian)
    if not os.path.exists(os.path.join(args.config_path, "preprocessing.yaml")):
        with open(os.path.join(args.config_path, "preprocessing.yaml"), 'w') as f:
            yaml.dump({
                'resize_dim': [64, 64], 'normalization_mean': [0.485,0.456,0.406], 'normalization_std': [0.229,0.224,0.225],
                'sobel_kernel_size': 3, 'gaussian_blur_kernel_size': [5,5], 'gaussian_blur_sigma': 1.0
            }, f)
    
    dummy_image_file = "temp_test_image_graph_ref.png"
    if args.image is None:
        cv2.imwrite(dummy_image_file, np.random.randint(0,255,(100,100,3),dtype=np.uint8))
        test_graph_pipeline(args.config_path, dummy_image_path=dummy_image_file)
        os.remove(dummy_image_file)
    else:
        if not os.path.exists(args.image):
            print(f"Error: Test image {args.image} not found.")
        else:
            test_graph_pipeline(args.config_path, dummy_image_path=args.image)