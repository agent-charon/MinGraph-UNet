import torch
import torch.nn.functional as F
import numpy as np

class PatchGraphConstructor:
    def __init__(self, patch_size=16):
        """
        Initializes the PatchGraphConstructor.

        Args:
            patch_size (int): The S_h and S_w size of each square patch.
        """
        self.patch_size = patch_size

    def image_to_patches(self, image_tensor_chw):
        """
        Converts an image tensor into a sequence of patches.
        Uses unfold for efficient patch extraction.

        Args:
            image_tensor_chw (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Patches tensor of shape (Num_Patches, C, Patch_H, Patch_W).
            tuple: (num_patches_h, num_patches_w)
        """
        C, H, W = image_tensor_chw.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            # Pad if not perfectly divisible
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            image_tensor_chw = F.pad(image_tensor_chw, (0, pad_w, 0, pad_h))
            C, H, W = image_tensor_chw.shape # New dimensions

        # Unfold the image into patches
        # .unfold(dimension, size, step)
        patches = image_tensor_chw.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        # patches shape: (C, num_patches_h, num_patches_w, patch_size, patch_size)
        
        num_patches_h = patches.shape[1]
        num_patches_w = patches.shape[2]

        # Permute and reshape to (num_patches_h * num_patches_w, C, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, C, self.patch_size, self.patch_size)
        
        return patches, (num_patches_h, num_patches_w)

    def construct_patch_graph(self, image_tensor_chw, patch_features_flat):
        """
        Constructs a graph where nodes are patches and edges connect spatially adjacent patches (4-connectivity).

        Args:
            image_tensor_chw (torch.Tensor): Original image tensor (C, H, W) - used for patch grid dimensions.
            patch_features_flat (torch.Tensor): Flattened features for each patch (Num_Patches, Feature_Dim).
                                                These are the node features for the graph.

        Returns:
            torch.Tensor: Node features (Num_Patches, Feature_Dim).
            torch.Tensor: Edge index (2, Num_Edges) in COO format (PyTorch Geometric style).
                          Row 0 contains source nodes, Row 1 contains target nodes.
        """
        C_img, H_img, W_img = image_tensor_chw.shape
        
        # Calculate number of patches in height and width
        # Account for potential padding if original image was not perfectly divisible
        num_patches_h = (H_img + self.patch_size -1) // self.patch_size 
        num_patches_w = (W_img + self.patch_size -1) // self.patch_size
        num_total_patches = num_patches_h * num_patches_w

        if patch_features_flat.shape[0] != num_total_patches:
            raise ValueError(f"Number of patch features ({patch_features_flat.shape[0]}) "
                             f"does not match expected number of patches ({num_total_patches}) "
                             f"for image {H_img}x{W_img} and patch size {self.patch_size}.")

        node_features = patch_features_flat # Already provided

        # Create edges (4-connectivity)
        adj = []
        for r in range(num_patches_h):
            for c in range(num_patches_w):
                node_idx = r * num_patches_w + c
                # Right neighbor
                if c + 1 < num_patches_w:
                    neighbor_idx = r * num_patches_w + (c + 1)
                    adj.append([node_idx, neighbor_idx])
                    adj.append([neighbor_idx, node_idx]) # Add symmetric edge
                # Down neighbor
                if r + 1 < num_patches_h:
                    neighbor_idx = (r + 1) * num_patches_w + c
                    adj.append([node_idx, neighbor_idx])
                    adj.append([neighbor_idx, node_idx]) # Add symmetric edge
        
        if not adj: # Handle single patch case or if no edges are formed
            edge_index = torch.empty((2, 0), dtype=torch.long, device=patch_features_flat.device)
        else:
            edge_index = torch.tensor(adj, dtype=torch.long, device=patch_features_flat.device).t().contiguous()
            # Remove duplicate edges (if any resulted from symmetric addition logic, though current logic avoids it for directed pairs)
            # For undirected graph GAT, typically you want both (i,j) and (j,i)
            # edge_index = torch.unique(edge_index, dim=1) # Not needed if added symmetrically one by one.

        return node_features, edge_index

    def get_patch_features_from_unet_encoder(self, unet_encoder_features, patches_coords_info):
        """
        Extracts features for each patch by averaging U-Net encoder features within the patch boundaries.
        This is one way to get initial patch features. The paper says:
        "The features of each node are extracted using the initial layers of the U-Net encoder,
         which captures local spatial information."
        Alternatively, if patches_tensor (C,P_H,P_W) is available, one could feed each patch
        through a small convnet, or use the U-Net features corresponding to the patch region.

        Args:
            unet_encoder_features (torch.Tensor): Feature map from U-Net encoder (e.g., B, F, H_feat, W_feat).
                                                  Assuming batch size 1 for simplicity here.
            patches_coords_info (tuple): (num_patches_h, num_patches_w)
                                         and implicitly the original image H,W and patch_size are known.
            original_image_h_w (tuple): (H_orig, W_orig) of the image input to U-Net
        
        Returns:
            torch.Tensor: (Num_Patches, Feature_Dim)
        """
        # This method requires careful handling of spatial correspondence between
        # U-Net feature map resolution and original image patch locations.
        # A simpler approach: take the extracted patches (image_to_patches)
        # and pass them through a small feature extractor (e.g., a few conv layers from U-Net encoder).
        # For now, let's assume `patch_features_flat` is already computed externally
        # using U-Net features and then enhanced by sobel, hist_eq, etc.
        # The paper says: "These edge features are concatenated with the U-Net encoder features
        # to provide additional context for the graph nodes." (Sec 3.2.3)

        # This function would be more complex and depend on U-Net architecture details.
        # We'll assume for now that a (Num_Patches, U_Net_Feature_Dim) tensor is obtained
        # and then other processed image features (Sobel, etc., per patch) are concatenated.
        raise NotImplementedError("Direct extraction from U-Net feature map to patch average needs careful impl. "
                                  "Consider processing patches or using global average pool per patch region.")


if __name__ == '__main__':
    constructor = PatchGraphConstructor(patch_size=32)
    
    # Dummy image tensor (C, H, W)
    C, H, W = 3, 128, 128
    dummy_image = torch.randn(C, H, W)

    patches_tensor, (nph, npw) = constructor.image_to_patches(dummy_image)
    num_patches = patches_tensor.shape[0]
    print(f"Image shape: ({C}, {H}, {W})")
    print(f"Patch size: {constructor.patch_size}")
    print(f"Number of patches: {num_patches} ({nph} x {npw})")
    print(f"Patches tensor shape: {patches_tensor.shape}") # (Num_Patches, C, Patch_H, Patch_W)

    # Dummy patch features (e.g., after U-Net encoder and other processing)
    # Suppose each patch feature is a vector of size 64 after processing
    feature_dim = 64
    # For each patch, we could average its pixel values or pass it through a small CNN
    # Here, using average of patch pixels as a placeholder feature
    dummy_patch_features = patches_tensor.mean(dim=[1,2,3]).unsqueeze(1).repeat(1, feature_dim) # (Num_Patches, Feature_Dim)
    # A more realistic scenario:
    # 1. Get patches_tensor (N, C, P_H, P_W)
    # 2. Pass through a feature_extractor_cnn (e.g. part of U-Net encoder) -> (N, F_unet)
    # 3. Get Sobel, HistEq for each patch, flatten/pool -> (N, F_sobel), (N, F_histeq)
    # 4. Concatenate: dummy_patch_features = torch.cat([unet_patch_feats, sobel_patch_feats, ...], dim=1)

    print(f"Dummy patch features shape: {dummy_patch_features.shape}")

    node_feats, edge_idx = constructor.construct_patch_graph(dummy_image, dummy_patch_features)
    print(f"Node features shape: {node_feats.shape}") # (Num_Patches, Feature_Dim)
    print(f"Edge index shape: {edge_idx.shape}")   # (2, Num_Edges)
    print(f"Number of edges: {edge_idx.shape[1]}")

    # Test with non-divisible image size
    H_new, W_new = 130, 140
    dummy_image_non_div = torch.randn(C, H_new, W_new)
    patches_non_div, (nph_nd, npw_nd) = constructor.image_to_patches(dummy_image_non_div)
    num_patches_nd = patches_non_div.shape[0]
    print(f"\nNon-divisible image shape: ({C}, {H_new}, {W_new})")
    print(f"Padded to: ({patches_non_div.shape[2]*nph_nd}, {patches_non_div.shape[3]*npw_nd}) internally by unfold effectively")
    print(f"Number of patches (non-div): {num_patches_nd} ({nph_nd} x {npw_nd})")
    print(f"Patches tensor shape (non-div): {patches_non_div.shape}")

    dummy_patch_features_nd = patches_non_div.mean(dim=[1,2,3]).unsqueeze(1).repeat(1, feature_dim)
    node_feats_nd, edge_idx_nd = constructor.construct_patch_graph(dummy_image_non_div, dummy_patch_features_nd)
    print(f"Node features shape (non-div): {node_feats_nd.shape}")
    print(f"Edge index shape (non-div): {edge_idx_nd.shape}")
    print(f"Number of edges (non-div): {edge_idx_nd.shape[1]}")