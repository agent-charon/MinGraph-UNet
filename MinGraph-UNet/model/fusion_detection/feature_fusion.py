import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, unet_feature_dims, gat_feature_dim, fusion_method="concat"):
        """
        Fuses U-Net features (F_u) and GAT-refined embeddings (F_g).
        F_u can be multi-scale. F_g is typically from region-level GAT.

        Args:
            unet_feature_dims (list of int): List of channel dimensions for each U-Net scale to be fused.
                                             These features need to be spatially aligned (e.g., to a common resolution)
                                             before fusion if they come from different U-Net decoder stages.
            gat_feature_dim (int): Dimension of GAT embeddings (F_g).
                                   F_g is likely per-region, so it needs to be broadcasted/mapped to pixel/patch level.
            fusion_method (str): "concat" or potentially "add", "multiply".
        """
        super().__init__()
        self.unet_feature_dims = unet_feature_dims
        self.gat_feature_dim = gat_feature_dim
        self.fusion_method = fusion_method.lower()

        # Total dimension after U-Net features are processed and concatenated (if multi-scale)
        # This depends on how F_u is prepared. If F_u are from different scales,
        # they need to be resized to a common H,W and then concatenated channel-wise.
        # Let's assume F_u passed to forward() is already a single tensor (B, C_unet_total, H, W).
        # Or, if F_g is per-patch/pixel, then F_u and F_g need to be at the same resolution.

        # The paper says (Sec 3.6): "The refined embeddings from the GAT layers (F_g) are fused with
        # U-Net features F_u at multiple scales".
        # "F_f = Concat(F_u, F_g)"
        # This implies F_u and F_g must be spatially aligned and then concatenated.
        # If F_g are region embeddings, they must be "unpooled" or broadcast to pixels within their region.

        # We'll assume F_u and F_g provided to forward() are already spatially aligned.
        # For "concat", no specific layers are needed here, just torch.cat.
        # If dimensions need to be matched before "add", linear layers might be needed.
        
        # Example: if F_u is (B, C1, H, W) and F_g is (B, C2, H, W),
        # concat gives (B, C1+C2, H, W).

    def forward(self, f_u_list, f_g, target_spatial_size=None, region_to_pixel_map=None):
        """
        Args:
            f_u_list (list of torch.Tensor): List of U-Net feature maps [ (B, C_i, H_i, W_i), ... ].
                                           These are F_u from different scales.
            f_g (torch.Tensor): GAT-refined embeddings.
                                If per-region: (Num_total_regions_in_batch, D_gat).
                                If already per-pixel/patch: (B, D_gat, H_target, W_target).
            target_spatial_size (tuple, optional): (H_out, W_out). If provided, all F_u features
                                                   are resized to this common spatial dimension.
                                                   If None, uses the spatial size of the first F_u tensor.
            region_to_pixel_map (torch.Tensor, optional): (B, H_target, W_target) tensor where each pixel
                                                          value is the index of the region it belongs to.
                                                          Required if f_g is per-region. Max value should be
                                                          Num_total_regions_in_batch - 1.

        Returns:
            torch.Tensor: Fused features F_f (B, C_fused, H_out, W_out).
        """
        B = f_u_list[0].size(0)

        if target_spatial_size is None:
            target_spatial_size = (f_u_list[0].size(2), f_u_list[0].size(3)) # H, W

        processed_f_u = []
        for f_u_scale in f_u_list:
            if (f_u_scale.size(2), f_u_scale.size(3)) != target_spatial_size:
                f_u_resized = F.interpolate(f_u_scale, size=target_spatial_size, mode='bilinear', align_corners=False)
                processed_f_u.append(f_u_resized)
            else:
                processed_f_u.append(f_u_scale)
        
        f_u_combined = torch.cat(processed_f_u, dim=1) # (B, C_unet_total, H_out, W_out)

        # Process F_g: If f_g is per-region, it needs to be mapped to pixel/patch level.
        # f_g: (Num_total_regions_in_batch, D_gat)
        # region_to_pixel_map: (B, H_out, W_out) with region indices
        
        if f_g.ndim == 2 and region_to_pixel_map is not None: # f_g is per-region
            # Num_total_regions_in_batch should match f_g.shape[0]
            # D_gat = f_g.shape[1]
            # Create an empty tensor for pixel-level GAT features
            f_g_pixel = torch.zeros(B, self.gat_feature_dim, target_spatial_size[0], target_spatial_size[1], device=f_g.device)
            
            for b_idx in range(B):
                # Get region indices for this batch item
                # region_map_item = region_to_pixel_map[b_idx] # (H_out, W_out)
                # Get GAT features corresponding to these regions
                # This needs a mapping from global region index in f_g to per-batch region map values.
                # This part is tricky: `f_g` has features for *all* regions across the batch.
                # `region_to_pixel_map` should contain indices that map correctly into `f_g`.
                
                # Efficient way: use f_g as an embedding table
                # region_to_pixel_map must contain valid indices into f_g
                # Flatten region_map and gather:
                # map_flat = region_to_pixel_map[b_idx].view(-1) # (H*W)
                # gathered_feats = f_g[map_flat] # (H*W, D_gat)
                # f_g_pixel[b_idx] = gathered_feats.view(target_spatial_size[0], target_spatial_size[1], self.gat_feature_dim).permute(2,0,1)

                # Simpler if region_to_pixel_map already has appropriate global indices:
                # This is an advanced gather operation.
                # Assume region_to_pixel_map values are direct indices into f_g.
                # This is slow if looping:
                # for r_idx in range(target_spatial_size[0]):
                #     for c_idx in range(target_spatial_size[1]):
                #         region_id = region_to_pixel_map[b_idx, r_idx, c_idx].long()
                #         if region_id >= 0 and region_id < f_g.shape[0]: # Ensure valid index
                #             f_g_pixel[b_idx, :, r_idx, c_idx] = f_g[region_id]
                #         # else: handle background pixels or invalid indices (e.g. leave as zero)
                
                # Vectorized approach:
                # Create a meshgrid of pixel coordinates
                # Create a flat list of region indices for the current batch item
                current_region_map_flat = region_to_pixel_map[b_idx].view(-1).long() # (H*W)
                
                # Filter out invalid indices (e.g., if -1 used for background not in f_g)
                valid_mask = (current_region_map_flat >= 0) & (current_region_map_flat < f_g.shape[0])
                valid_indices_flat = current_region_map_flat[valid_mask]
                
                if valid_indices_flat.numel() > 0:
                    # Gather features for valid regions
                    gathered_gat_features = f_g[valid_indices_flat] # (Num_valid_pixels, D_gat)
                
                    # Place them into the f_g_pixel tensor
                    # This requires knowing where the valid_mask pixels were.
                    # Create a temporary full-size tensor for this item
                    temp_f_g_item_pixel = torch.zeros(self.gat_feature_dim, target_spatial_size[0] * target_spatial_size[1], device=f_g.device)
                    
                    # Scatter:
                    # valid_mask is (H*W). Convert to indices for scattering
                    pixel_indices_for_scatter = torch.arange(target_spatial_size[0] * target_spatial_size[1], device=f_g.device)[valid_mask]
                    if pixel_indices_for_scatter.numel() > 0 and gathered_gat_features.numel() > 0:
                         temp_f_g_item_pixel[:, pixel_indices_for_scatter] = gathered_gat_features.T # (D_gat, Num_valid_pixels)
                    
                    f_g_pixel[b_idx] = temp_f_g_item_pixel.view(self.gat_feature_dim, target_spatial_size[0], target_spatial_size[1])
                # Pixels not belonging to any valid region in f_g remain zero.
            
            f_g_aligned = f_g_pixel # (B, D_gat, H_out, W_out)

        elif f_g.ndim == 4: # f_g is already per-pixel/patch e.g. (B, D_gat, H, W)
            if (f_g.size(2), f_g.size(3)) != target_spatial_size:
                f_g_aligned = F.interpolate(f_g, size=target_spatial_size, mode='bilinear', align_corners=False)
            else:
                f_g_aligned = f_g
        else:
            raise ValueError(f"f_g has unsupported shape {f_g.shape}. "
                             "Expected (Num_regions, D_gat) with region_map or (B, D_gat, H, W).")

        # Fuse
        if self.fusion_method == "concat":
            f_fused = torch.cat([f_u_combined, f_g_aligned], dim=1)
        elif self.fusion_method == "add":
            # Dimensions must match for add. May require linear layers on f_u or f_g.
            if f_u_combined.shape[1] != f_g_aligned.shape[1]:
                raise ValueError("Channel dimensions must match for 'add' fusion or implement adaptation.")
            f_fused = f_u_combined + f_g_aligned
        else:
            raise NotImplementedError(f"Fusion method '{self.fusion_method}' not implemented.")
            
        return f_fused


if __name__ == '__main__':
    B = 2
    H, W = 32, 32 # Target spatial size for fusion
    
    # Dummy F_u (U-Net features from 2 scales)
    f_u1 = torch.randn(B, 64, H, W)      # Already at target size
    f_u2 = torch.randn(B, 128, H//2, W//2) # Needs upsampling
    f_u_list_test = [f_u1, f_u2]
    unet_dims = [f_u1.shape[1], f_u2.shape[1]]

    D_gat_test = 32

    # --- Test 1: F_g is already per-pixel ---
    print("--- Test 1: F_g is per-pixel ---")
    f_g_pixel_test = torch.randn(B, D_gat_test, H, W)
    fusion_module1 = FeatureFusion(unet_dims, D_gat_test, fusion_method="concat")
    fused_output1 = fusion_module1(f_u_list_test, f_g_pixel_test, target_spatial_size=(H,W))
    print(f"Fused output shape (F_g per-pixel): {fused_output1.shape}")
    # Expected channels: 64 (f_u1) + 128 (f_u2 resized) + 32 (f_g) = 224
    # But f_u_list is concatenated first, so 64+128 = 192 for f_u_combined.
    # Then 192 + 32 (f_g) = 224.
    # Correct: (sum of channels in f_u_list_test after resizing) + D_gat_test
    expected_channels1 = f_u1.shape[1] + f_u2.shape[1] + D_gat_test # No, f_u are combined first
    expected_channels1_actual = f_u_list_test[0].shape[1] + f_u_list_test[1].shape[1] + f_g_pixel_test.shape[1]
    
    # The f_u_list is concatenated to form f_u_combined first.
    # So f_u_combined has f_u1.C + f_u2.C channels.
    # Then f_g_aligned is concatenated.
    # Total channels = (f_u1.C + f_u2.C) + f_g.C
    
    C_f_u_combined = sum(f.shape[1] for f in f_u_list_test)
    expected_total_channels = C_f_u_combined + D_gat_test
    assert fused_output1.shape == (B, expected_total_channels, H, W)


    # --- Test 2: F_g is per-region ---
    print("\n--- Test 2: F_g is per-region ---")
    num_regions_batch0 = 3
    num_regions_batch1 = 2
    total_regions_in_batch = num_regions_batch0 + num_regions_batch1 # Example: 5 regions total
    
    f_g_region_test = torch.randn(total_regions_in_batch, D_gat_test) # (TotalRegions, D_gat)
    
    # Dummy region_to_pixel_map (B, H, W)
    # Values are indices into f_g_region_test
    region_map = torch.zeros(B, H, W, dtype=torch.long)
    # Batch 0: 3 regions
    region_map[0, :H//2, :W//2] = 0 # Region 0
    region_map[0, H//2:, :W//2] = 1 # Region 1
    region_map[0, :, W//2:] = 2     # Region 2
    # Batch 1: 2 regions (indices continue from batch 0 for f_g_region_test)
    region_map[1, :, :W//3] = 3     # Region 3 (mapped from f_g_region_test[3])
    region_map[1, :, W//3:] = 4     # Region 4 (mapped from f_g_region_test[4])
    # Add some background pixels (e.g. index -1, not mapped)
    region_map[0, 0, 0] = -1 


    fusion_module2 = FeatureFusion(unet_dims, D_gat_test, fusion_method="concat")
    fused_output2 = fusion_module2(f_u_list_test, f_g_region_test, 
                                   target_spatial_size=(H,W), 
                                   region_to_pixel_map=region_map)
    print(f"Fused output shape (F_g per-region): {fused_output2.shape}")
    assert fused_output2.shape == (B, expected_total_channels, H, W)
    # Check if background pixel in f_g_pixel part is zero
    print(f"Value at background pixel (0,0,0) in F_g part of fused_output2: {fused_output2[0, C_f_u_combined:, 0, 0].sum()}")
    # This depends on the gather logic; if -1 is not handled, it might error or misbehave.
    # My implementation should leave it as zero.