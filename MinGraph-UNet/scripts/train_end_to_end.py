import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2 # For image ops if needed outside preprocessor

# Project structure imports
# Assuming scripts/ is run from the project root, or paths are adjusted
# For development, you might add project root to sys.path
# import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(script_dir, '..'))
# if project_root not in sys.path:
#    sys.path.insert(0, project_root)

from model.unet.unet_model import UNet
from model.unet.shape_loss import EllipticalShapeLoss
from model.unet.feature_loss import FeatureConsistencyLoss
from model.gat.graph_attention import GATNetwork
from model.graph_partition.mincut_refinement import MinCutRefinement
from model.fusion_detection.feature_fusion import FeatureFusion
from model.fusion_detection.detection_head import DetectionHead

from preprocessing.image_preprocessing.image_preprocess import ImagePreprocessor
from preprocessing.graph_construction.patch_graph_construction import PatchGraphConstructor
from preprocessing.graph_feature_processing.edge_detection import EdgeDetector
from preprocessing.graph_feature_processing.histogram_equalization import HistogramEqualizer
# from preprocessing.graph_feature_processing.gaussian_smoothing import GaussianSmoother # Not used in patch feature loop in graph_ref_test

from utils.mango_dataset import MangoDataset # Use the new dataset file

# --- Helper: Segment Predictor for MinCut ---
# This network predicts soft assignments of patches to K segments for Ncut loss
class PatchSegmentPredictor(nn.Module):
    def __init__(self, in_dim, num_segments, hidden_dim=None, use_gnn=False, num_gnn_layers=1, num_heads=1):
        super().__init__()
        self.use_gnn = use_gnn
        if use_gnn:
            # Using a GAT layer to predict segment logits based on graph structure
            self.gnn_predictor = GATNetwork(
                node_feature_dim=in_dim,
                hidden_dim=hidden_dim if hidden_dim else in_dim, # GAT hidden dim
                output_dim=num_segments, # GAT output dim = num_segments
                num_heads=num_heads,
                num_gat_layers=num_gnn_layers, # Can be simple 1-layer GAT
                dropout_rate=0.1, # Example
                alpha=0.2       # Example
            )
        else:
            # Simple MLP predictor
            if hidden_dim is None: hidden_dim = in_dim * 2
            self.mlp_predictor = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_segments)
            )
    
    def forward(self, x, edge_index=None):
        if self.use_gnn:
            if edge_index is None:
                raise ValueError("edge_index must be provided for GNN-based segment predictor.")
            return self.gnn_predictor(x, edge_index)
        else:
            return self.mlp_predictor(x)

# --- Helper: Total Variation Loss ---
class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        # x is expected to be (B, C, H, W), e.g., a segmentation probability map for one class
        # or the logits if we want to smooth logits.
        # Paper implies L_smooth on "segmentation mask S_i,j" (Sec 3.6, after Eq 10)
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return self.weight * (h_tv / count_h + w_tv / count_w) / batch_size


def load_config(config_dir, config_name):
    with open(os.path.join(config_dir, config_name), 'r') as f:
        return yaml.safe_load(f)

def get_config_recursively(cfg_dict, key_path, default=None):
    current_level = cfg_dict
    for key_part in key_path.split('.'):
        if isinstance(current_level, dict) and key_part in current_level:
            current_level = current_level[key_part]
        else:
            return default
    return current_level

def train_end_to_end(config_path):
    # --- Configuration Loading ---
    dataset_cfg = load_config(config_path, "dataset.yaml")
    model_cfg = load_config(config_path, "model.yaml")
    preproc_cfg = load_config(config_path, "preprocessing.yaml")
    train_cfg = load_config(config_path, "training.yaml")

    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Preprocessing Modules Init ---
    image_preprocessor = ImagePreprocessor(
        resize_dim=tuple(preproc_cfg['resize_dim']),
        mean=tuple(preproc_cfg['normalization_mean']),
        std=tuple(preproc_cfg['normalization_std']),
        apply_augmentation=True
    )
    patch_constructor = PatchGraphConstructor(patch_size=get_config_recursively(model_cfg, 'graph_construction.patch_size', 16))
    edge_detector = EdgeDetector(kernel_size=get_config_recursively(preproc_cfg, 'sobel_kernel_size', 3))
    hist_equalizer = HistogramEqualizer()

    # --- Model Initialization ---
    # 1. U-Net
    unet_config = get_config_recursively(model_cfg, 'unet')
    unet_model = UNet(
        in_channels=unet_config['in_channels'],
        num_classes=unet_config['out_channels'], # For initial segmentation
        init_features=unet_config['init_features'],
        depth=unet_config['depth']
    ).to(device)

    # 2. Patch GAT (1st GAT stage)
    patch_gat_cfg = get_config_recursively(model_cfg, 'gat') # Assuming one GAT config for both stages for now
    # Input dim for patch GAT needs to be calculated based on concatenated patch features
    # Placeholder: U-Net patch feat dim + Sobel patch feat dim (1) + HistEq patch feat dim (3 for RGB)
    # This needs to be determined dynamically or configured. Let's set a placeholder value.
    dummy_unet_patch_feat_dim = 16 # Example
    patch_feat_concat_dim = dummy_unet_patch_feat_dim + 1 + 3 # Placeholder
    
    patch_gat_model = GATNetwork(
        node_feature_dim=patch_feat_concat_dim, # This needs to be accurate
        hidden_dim=patch_gat_cfg['hidden_dim'],
        output_dim=patch_gat_cfg['output_dim'], # Output for segment predictor
        num_heads=patch_gat_cfg['num_heads'],
        num_gat_layers=1, # As per Figure 1 and common practice for this stage
        dropout_rate=patch_gat_cfg['dropout'],
        alpha=patch_gat_cfg['alpha']
    ).to(device)
    
    # 3. MinCut Refinement (Segment Predictor part)
    num_segments_for_ncut = get_config_recursively(dataset_cfg, 'num_semantic_regions', unet_config['out_channels'])
    segment_predictor_network = PatchSegmentPredictor(
        in_dim=patch_gat_cfg['output_dim'], # Input from patch_gat
        num_segments=num_segments_for_ncut,
        hidden_dim=patch_gat_cfg['output_dim'] // 2, # Example hidden dim
        use_gnn=True, # Can be MLP or GNN
        num_gnn_layers=1,
        num_heads=max(1,patch_gat_cfg['num_heads']//2)
    ).to(device)
    mincut_module = MinCutRefinement() # Ncut loss calculation part

    # 4. Region GAT (2nd GAT stage)
    # Input dim for region GAT: features aggregated from U-Net/patches for each region
    # Output dim can be same as patch_gat_cfg['output_dim'] or different.
    # For simplicity, assume region features are derived to match patch_gat_cfg['output_dim'].
    region_gat_model = GATNetwork(
        node_feature_dim=patch_gat_cfg['output_dim'], # Example, needs careful consideration
        hidden_dim=patch_gat_cfg['hidden_dim'],
        output_dim=patch_gat_cfg['output_dim'], # F_g dimension
        num_heads=patch_gat_cfg['num_heads'],
        num_gat_layers=1,
        dropout_rate=patch_gat_cfg['dropout'],
        alpha=patch_gat_cfg['alpha']
    ).to(device)

    # 5. Feature Fusion
    # U-Net features F_u (e.g. from decoder). Need their channel dimensions.
    # If using all decoder stages from U-Net (depth=4, init_feat=32): F_u dims = [32, 64, 128, 256]
    # This needs to align with UNetDecoder's output.
    # Let's assume we take the shallowest decoder output (highest res) for F_u for now.
    # Or configure which F_u scales to use.
    f_u_channels_for_fusion = [unet_config['init_features']] # Example: shallowest decoder output
    
    fusion_cfg = get_config_recursively(model_cfg, 'fusion_detection', {})
    feature_fuser = FeatureFusion(
        unet_feature_dims=f_u_channels_for_fusion, # List of channel dims for F_u scales
        gat_feature_dim=patch_gat_cfg['output_dim'] # Dimension of F_g
    ).to(device)

    # 6. Detection Head
    # Input channels to detection head is output of feature_fuser
    fused_feat_channels = sum(f_u_channels_for_fusion) + patch_gat_cfg['output_dim']
    detection_head = DetectionHead(
        in_features_channels=fused_feat_channels,
        num_classes=get_config_recursively(dataset_cfg, 'num_detection_classes', 1), # e.g., 1 for mango
        fc_hidden_dim=get_config_recursively(fusion_cfg, 'fc_hidden_dim', 256)
    ).to(device)

    # --- Loss Functions ---
    losses_cfg = get_config_recursively(model_cfg, 'losses')
    l_shape_loss_fn = EllipticalShapeLoss().to(device)
    l_feature_loss_fn = FeatureConsistencyLoss(margin=losses_cfg.get('feature_loss_margin', 1.0)).to(device)
    # L_partition is calculated by mincut_module
    l_smooth_loss_fn = TVLoss().to(device) # For final segmentation map from F_f
    
    # Standard segmentation loss for initial U-Net output
    unet_segmentation_criterion = nn.CrossEntropyLoss().to(device)
    # Detection losses (e.g., L1 for bbox, BCE for confidence/class) would be separate
    # For simplicity, paper's L_total doesn't explicitly list detection losses,
    # but they are implied if 'b' and 's' are trained.
    # Let's assume for now detection is a final step and L_total focuses on seg refinement.

    # --- Optimizer ---
    all_trainable_params = \
        list(unet_model.parameters()) + \
        list(patch_gat_model.parameters()) + \
        list(segment_predictor_network.parameters()) + \
        list(region_gat_model.parameters()) + \
        list(feature_fuser.parameters()) + \
        list(detection_head.parameters())
        
    optimizer_name = train_cfg['optimizer'].lower()
    if optimizer_name == "adam":
        optimizer = optim.Adam(all_trainable_params, lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(all_train_params, lr=train_cfg['learning_rate'], momentum=train_cfg['sgd_momentum'], weight_decay=train_cfg['weight_decay'])
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    # LR Scheduler
    scheduler = None
    if train_cfg.get('lr_scheduler'):
        if train_cfg['lr_scheduler'].lower() == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['lr_gamma'])

    # --- Dataset and DataLoader ---
    train_dataset_path = os.path.join(dataset_cfg['data_root'], dataset_cfg['train_dir'])
    train_dataset = MangoDataset(
        image_dir=os.path.join(train_dataset_path, dataset_cfg['image_folder']),
        mask_dir=os.path.join(train_dataset_path, dataset_cfg['mask_folder']),
        preprocessor=image_preprocessor,
        num_classes=unet_config['out_channels']
    )
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'], drop_last=True)

    # --- Training Loop ---
    print("Starting End-to-End MinGraph-UNet Training...")
    for epoch in range(train_cfg['num_epochs']):
        # Set all models to train mode
        unet_model.train(); patch_gat_model.train(); segment_predictor_network.train()
        region_gat_model.train(); feature_fuser.train(); detection_head.train()
        
        running_losses = {
            'total': 0.0, 'l_unet_seg': 0.0, 'l_shape': 0.0, 'l_feature': 0.0,
            'l_partition': 0.0, 'l_smooth': 0.0 # Add detection losses if any
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        for batch_idx, (images_batch, gt_masks_batch) in enumerate(progress_bar):
            images_batch = images_batch.to(device) # (B, C, H_orig, W_orig)
            gt_masks_batch = gt_masks_batch.to(device) # (B, H_orig, W_orig) - semantic ground truth
            
            batch_size_current = images_batch.size(0)
            optimizer.zero_grad()

            # === Stage 1: U-Net ===
            # initial_seg_logits: (B, NumClasses, H, W)
            # encoder_skips: list of (B, C, H_s, W_s)
            # decoder_features_Fu: list of (B, C_d, H_d, W_d) - shallow to deep
            initial_seg_logits, encoder_skips, decoder_features_Fu = unet_model(images_batch)
            
            # Loss for initial U-Net segmentation (e.g. CrossEntropy)
            loss_unet_seg = unet_segmentation_criterion(initial_seg_logits, gt_masks_batch)

            # L_shape (Elliptical Shape Loss)
            # Requires instance masks. For semantic gt_masks_batch, this is tricky.
            # If gt_masks_batch is semantic, L_shape might need instance proposals from initial_seg_logits
            # or operate on the entire foreground class from gt_masks_batch.
            # For now, simplified: use GT mask for foreground class if binary.
            # This needs `object_masks_list` from GT. For simplicity, let's skip for now or use placeholder.
            # object_gt_masks_list = ... # derive from gt_masks_batch (B, List[obj_mask_HW])
            # loss_shape = l_shape_loss_fn(torch.softmax(initial_seg_logits, dim=1), object_gt_masks_list)
            loss_shape = torch.tensor(0.0, device=device) # Placeholder

            # --- Graph Processing Loop (per image in batch, as graph ops are often per-sample) ---
            # This loop is a simplification. Batching graph operations requires libraries like PyG.
            # For now, accumulate losses and average.
            batch_loss_feature = 0.0
            batch_loss_partition = 0.0
            
            # Store features for batch-level fusion later
            batch_f_g_pixel_mapped_list = []
            batch_f_u_for_fusion_list = []


            for i in range(batch_size_current):
                img_tensor_single = images_batch[i] # (C,H,W)
                initial_seg_logits_single = initial_seg_logits[i] # (NumClasses, H,W)
                # F_u for this image (e.g. shallowest decoder output)
                # decoder_features_Fu is list (depth) of (B,C,H,W). We need item `i` from batch.
                f_u_single_list = [feat[i] for feat in decoder_features_Fu] # List of (C,H,W)

                # === Stage 2: Patch Graph Construction & Initial Features ===
                # Get RGB image before ToTensor normalization for Sobel/HistEq
                # This requires undoing normalization or passing original resized image.
                # For simplicity in loop, assume img_tensor_single can be used to get patches for structure,
                # and features are derived. A more proper way would be to process before ToTensor.
                
                # Placeholder patch features (as in graph_refinement.py test)
                # This part needs careful implementation of actual feature extraction per patch.
                # For now, using dummy placeholder features.
                
                # 1. Construct Patches (structure)
                patches_struct, (nph, npw) = patch_constructor.image_to_patches(img_tensor_single)
                num_patches_single = patches_struct.shape[0]

                # 2. Create initial patch features (concat U-Net patch feat + Sobel patch feat + HistEq patch feat)
                #    This requires running U-Net encoder on patches OR mapping encoder_skips to patches
                #    AND running Sobel/HistEq on patches of original image.
                #    This is complex. Let's use a simplified feature vector per patch.
                #    Assume `patch_feat_concat_dim` is known and we get `current_patch_features`
                current_patch_features = torch.randn(num_patches_single, patch_feat_concat_dim, device=device) # Placeholder

                # 3. Construct patch graph
                _, patch_edge_index_single = patch_constructor.construct_patch_graph(img_tensor_single, current_patch_features)
                
                # === Stage 3: Patch GAT ===
                gat_refined_patch_feats_single = patch_gat_model(current_patch_features, patch_edge_index_single)

                # === L_feature (Feature Consistency Loss) ===
                # Requires U-Net features per patch (f_unet_patches) and GAT features per patch (f_graph_patches)
                # And patch_region_labels_y (semantic label per patch from initial_seg_logits or GT)
                # f_unet_patches: (N_patches, D). Derive from initial_seg_logits or encoder_skips. Placeholder.
                f_unet_patches_placeholder = torch.randn_like(gat_refined_patch_feats_single)
                
                # patch_region_labels_y: from argmax of initial_seg_logits_single, pooled over patch regions
                # For simplicity, random labels for now
                pred_patch_labels_y = torch.randint(0, unet_config['out_channels'], (num_patches_single,), device=device)
                
                loss_feature_single = l_feature_loss_fn(f_unet_patches_placeholder, gat_refined_patch_feats_single, pred_patch_labels_y)
                batch_loss_feature += loss_feature_single

                # === Stage 4: MinCut Refinement (L_partition) ===
                loss_partition_single, s_star_soft_patches_single = mincut_module(
                    gat_refined_patch_feats_single, patch_edge_index_single,
                    num_segments_for_ncut, segment_predictor_network
                )
                batch_loss_partition += loss_partition_single
                
                # `s_star_soft_patches_single` is (N_patches, K_segments). This is patch-level refined seg.
                # Convert to hard labels for region graph construction
                hard_patch_labels_single = torch.argmax(s_star_soft_patches_single, dim=1) # (N_patches)

                # === Stage 5: Region Graph Construction & Region GAT ===
                # This is highly conceptual without a full region graph library.
                # 1. Identify unique regions from `hard_patch_labels_single`.
                # 2. For each region, aggregate features (e.g., from `gat_refined_patch_feats_single`).
                # 3. Construct region adjacency (e.g., if patches of different regions are neighbors).
                # For now, assume we get `region_node_features` and `region_edge_index`.
                
                # Placeholder: Treat each predicted segment as a 'region'.
                # If K_segments are predicted, we have K region nodes.
                # Region features: average GAT-refined patch features for patches in that segment.
                num_regions_found = num_segments_for_ncut # Simplification
                region_node_features_single = torch.zeros(num_regions_found, gat_refined_patch_feats_single.shape[1], device=device)
                for k_seg in range(num_regions_found):
                    mask_k = (hard_patch_labels_single == k_seg)
                    if mask_k.sum() > 0:
                        region_node_features_single[k_seg] = gat_refined_patch_feats_single[mask_k].mean(dim=0)
                
                # Placeholder region graph (e.g., fully connected for K regions)
                if num_regions_found > 1:
                    src_reg, tgt_reg = torch.triu_indices(num_regions_found, num_regions_found, offset=1)
                    region_edge_index_single = torch.stack([torch.cat([src_reg,tgt_reg]), torch.cat([tgt_reg,src_reg])], dim=0).to(device)
                else: # Single region, no edges
                    region_edge_index_single = torch.empty((2,0), dtype=torch.long, device=device)

                # Region GAT
                if num_regions_found > 0 and region_edge_index_single.numel() > 0 :
                    f_g_region_embeddings_single = region_gat_model(region_node_features_single, region_edge_index_single) # (K_regions, D_gat_out)
                elif num_regions_found > 0: # No edges, pass features through if GAT handles it (e.g. MLP fallback)
                    # Our GAT will likely fail with no edges. For simplicity, if no edges, use unrefined region features.
                    f_g_region_embeddings_single = region_node_features_single # Use aggregated features directly
                else: # No regions found
                    f_g_region_embeddings_single = torch.empty((0, patch_gat_cfg['output_dim']), device=device)


                # We need to map f_g_region_embeddings_single (per region) back to pixels/patches.
                # The `hard_patch_labels_single` (N_patches) gives the region_id for each patch.
                # So, f_g_patch_mapped_single = f_g_region_embeddings_single[hard_patch_labels_single] (N_patches, D_gat_out)
                # This then needs to be "unpatched" to pixel level for fusion with F_u. This is complex.

                # --- Simpler F_g for fusion: Use S*_soft_patches ---
                # Let F_g be derived from s_star_soft_patches (N_patches, K_segments)
                # by projecting K_segments to D_gat_out.
                # This bypasses explicit region graph for fusion step, but uses its output.
                # This is a deviation from paper if region GAT is strictly on region graph.
                # For now, assume f_g_patch_mapped_single is (N_patches, D_gat_out)
                if num_patches_single > 0 and f_g_region_embeddings_single.numel() > 0:
                    f_g_patch_mapped_single = f_g_region_embeddings_single[hard_patch_labels_single]
                else: # Handle cases with no patches or no region embeddings
                    f_g_patch_mapped_single = torch.zeros((num_patches_single, patch_gat_cfg['output_dim']), device=device)


                # Reshape f_g_patch_mapped from (N_p, D) to (D, H_p, W_p) where H_p, W_p are patch grid dims
                # Then "unpatch" to (D, H_img, W_img)
                f_g_pixel_mapped_single = f_g_patch_mapped_single.T.reshape(patch_gat_cfg['output_dim'], nph, npw) # (D, nph, npw)
                # Upsample this to original image resolution using patch_constructor logic reversed
                # This is effectively undoing the patching operation.
                # F.interpolate could achieve this by treating nph, npw as spatial dims.
                # Target size: H_orig, W_orig from img_tensor_single
                target_h, target_w = img_tensor_single.shape[1], img_tensor_single.shape[2]
                f_g_pixel_mapped_single = F.interpolate(
                    f_g_pixel_mapped_single.unsqueeze(0), # Add batch dim for interpolate
                    size=(target_h, target_w), 
                    mode='nearest' # Or bilinear for smoother features
                ).squeeze(0) # (D_gat_out, H_orig, W_orig)
                batch_f_g_pixel_mapped_list.append(f_g_pixel_mapped_single)

                # Store F_u for this item (e.g., shallowest decoder output)
                batch_f_u_for_fusion_list.append(f_u_single_list[0]) # Taking only the first (shallowest) F_u scale

            # Average losses from the per-image loop
            loss_feature = batch_loss_feature / batch_size_current
            loss_partition = batch_loss_partition / batch_size_current
            
            # Stack collected features for batch-wise fusion
            f_g_batch_pixel_mapped = torch.stack(batch_f_g_pixel_mapped_list, dim=0) # (B, D_gat, H, W)
            f_u_batch_for_fusion = torch.stack(batch_f_u_for_fusion_list, dim=0) # (B, C_fu, H, W)


            # === Stage 6: Feature Fusion ===
            # F_u for fusion: e.g. shallowest from decoder_features_Fu
            # F_g for fusion: from region GAT, mapped to pixel/patch level
            # The feature_fuser expects a list of F_u scales. Here we use one.
            f_fused_batch = feature_fuser(
                f_u_list=[f_u_batch_for_fusion], 
                f_g=f_g_batch_pixel_mapped,
                target_spatial_size=(f_u_batch_for_fusion.size(2), f_u_batch_for_fusion.size(3))
            ) # (B, C_fused, H, W)

            # === Stage 7: Detection Head ===
            # This part is simplified. Detection often has its own GT (bboxes) and losses.
            # Paper's L_total doesn't include detection losses.
            # For now, just forward pass. If training detection, add bbox/class GT and loss.
            if get_config_recursively(dataset_cfg, 'num_detection_classes', 1) > 1:
                 pred_bboxes, pred_confidence, pred_class_scores = detection_head(f_fused_batch)
            else:
                 pred_bboxes, pred_confidence = detection_head(f_fused_batch)
            # Detection losses would be calculated here if training for detection.

            # === L_smooth (Total Variation Loss) ===
            # Applied to a segmentation map derived from F_fused.
            # E.g., pass F_fused through a final conv to get seg logits, then TV on probs.
            # For simplicity, apply to confidence map if it's like a fg/bg score.
            # Or, if initial_seg_logits is considered the target for smoothness after refinement:
            # loss_smooth = l_smooth_loss_fn(torch.softmax(initial_seg_logits, dim=1)[:,1,:,:].unsqueeze(1)) # TV on prob of class 1
            loss_smooth = l_smooth_loss_fn(pred_confidence.view(batch_size_current, 1, 1, 1).expand(-1,-1,f_fused_batch.size(2),f_fused_batch.size(3))) # Dummy application

            # === Total Loss (as per paper's L_total formula components) ===
            lambda_shape = losses_cfg.get('l_shape_weight', 0.1)
            lambda_feature = losses_cfg.get('l_feature_weight', 0.1)
            lambda_partition = losses_cfg.get('l_partition_weight', 0.5)
            lambda_smooth = losses_cfg.get('l_smooth_weight', 0.2)

            # Add U-Net's own segmentation loss to the total, as it's foundational
            # This might be part of an implicit L_U-Net term.
            total_loss = (loss_unet_seg +
                          lambda_shape * loss_shape + 
                          lambda_feature * loss_feature + 
                          lambda_partition * loss_partition + 
                          lambda_smooth * loss_smooth)
            
            total_loss.backward()
            optimizer.step()

            running_losses['total'] += total_loss.item()
            running_losses['l_unet_seg'] += loss_unet_seg.item()
            running_losses['l_shape'] += loss_shape.item() if torch.is_tensor(loss_shape) else loss_shape
            running_losses['l_feature'] += loss_feature.item()
            running_losses['l_partition'] += loss_partition.item()
            running_losses['l_smooth'] += loss_smooth.item()

            postfix_stats = {k: v / (batch_idx + 1) for k, v in running_losses.items()}
            postfix_stats['lr'] = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(postfix_stats)

        epoch_avg_loss = running_losses['total'] / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Total Loss: {epoch_avg_loss:.4f}")
        for k, v in running_losses.items():
            if k != 'total': print(f"  Avg {k}: {v / len(train_loader):.4f}")

        if scheduler:
            scheduler.step()
        
        # Save checkpoint (simplified)
        if (epoch + 1) % train_cfg['save_epoch_interval'] == 0:
            # Save all model states
            # ...
            print(f"Checkpoint for epoch {epoch+1} would be saved here.")

    print("End-to-End Training Finished.")
    # Save final model
    # ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MinGraph-UNet End-to-End.")
    parser.add_argument('--config_path', type=str, default="configs/", help="Path to config files dir.")
    args = parser.parse_args()

    # Create dummy configs if not present (ensure all necessary keys are there for this complex script)
    # This will be more involved than for train_segmentation.py
    # Example: Ensure model_cfg.losses, model_cfg.fusion_detection etc. are present.
    # Due to complexity, users should ensure configs are well-defined.
    
    # Create dummy data for a quick test run
    base_dir_test = "temp_mango_dataset_e2e"
    train_img_dir_test = os.path.join(base_dir_test, "train", "images")
    train_mask_dir_test = os.path.join(base_dir_test, "train", "masks")
    if not os.path.exists(train_img_dir_test): os.makedirs(train_img_dir_test)
    if not os.path.exists(train_mask_dir_test): os.makedirs(train_mask_dir_test)
    for i in range(4): # Enough for batch_size=2 (from default training.yaml)
        cv2.imwrite(os.path.join(train_img_dir_test, f"img_{i}.png"), np.random.randint(0,255,(100,100,3), dtype=np.uint8))
        cv2.imwrite(os.path.join(train_mask_dir_test, f"img_{i}.png"), np.random.randint(0,2,(100,100), dtype=np.uint8))
    
    # Update dummy configs to ensure all paths are covered.
    # Minimal config setup for testing:
    if not os.path.exists(args.config_path): os.makedirs(args.config_path)
    with open(os.path.join(args.config_path, "dataset.yaml"), 'w') as f:
        yaml.dump({'data_root': base_dir_test, 'train_dir': 'train', 'image_folder': 'images', 'mask_folder': 'masks',
                   'num_classes': 2, 'num_semantic_regions': 2, 'num_detection_classes': 1}, f)
    with open(os.path.join(args.config_path, "model.yaml"), 'w') as f:
        yaml.dump({
            'unet': {'in_channels': 3, 'out_channels': 2, 'init_features': 8, 'depth': 1}, # Minimal
            'graph_construction': {'patch_size': 32}, # Match image size if depth=1, resize=64
            'gat': {'hidden_dim': 16, 'output_dim': 8, 'num_heads': 1, 'dropout': 0.1, 'alpha': 0.2},
            'mincut': {}, # Default params for MinCutRefinement
            'fusion_detection': {'fc_hidden_dim': 32},
            'losses': {'l_shape_weight':0.0, 'l_feature_weight':0.1, 'l_partition_weight':0.1, 'l_smooth_weight':0.01, 'feature_loss_margin':1.0}
        }, f)
    with open(os.path.join(args.config_path, "preprocessing.yaml"), 'w') as f:
        yaml.dump({'resize_dim': [64, 64], 'normalization_mean': [0.485,0.456,0.406], 'normalization_std': [0.229,0.224,0.225],
                   'sobel_kernel_size':3, 'gaussian_blur_kernel_size':[3,3], 'gaussian_blur_sigma':1.0},f)
    with open(os.path.join(args.config_path, "training.yaml"), 'w') as f:
        yaml.dump({'device':'cpu', 'batch_size':1, 'num_epochs':1, 'learning_rate':1e-4, 'optimizer':'Adam', 'weight_decay':1e-5,
                   'save_epoch_interval':1, 'checkpoint_dir': 'outputs/checkpoints_e2e_test'}, f)

    print("Starting E2E training script test with dummy data and configs...")
    try:
        train_end_to_end(args.config_path)
    except Exception as e:
        print(f"ERROR during E2E training script test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        if os.path.exists(base_dir_test): shutil.rmtree(base_dir_test)
        if os.path.exists("outputs/checkpoints_e2e_test"): shutil.rmtree("outputs/checkpoints_e2e_test")