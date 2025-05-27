import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureConsistencyLoss(nn.Module):
    """
    L_feature = Σ Σ [y_ij ||F_U-Net,i - F_graph,j||^2 + (1 - y_ij) max(0, m - ||F_U-Net,i - F_graph,j||)^2]
    where y_ij is 1 if pixel/patch i (from U-Net) and patch/node j (from graph) correspond
    to the same region, 0 otherwise. m is a margin.

    This loss aims to make U-Net features and Graph features consistent for corresponding regions.
    It implies a correspondence mapping between U-Net pixels/regions and graph nodes (patches).
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, f_unet, f_graph, correspondence_map_y, regions_unet=None, regions_graph=None):
        """
        Args:
            f_unet (torch.Tensor): Features from U-Net. Shape could be (B, Num_Unet_Items, FeatureDim)
                                   or (B, C, H, W) then processed.
                                   "U-Net features F_l from multiple stages" (Sec 3.3)
                                   "F_U-Net,i"
            f_graph (torch.Tensor): Features from graph nodes. Shape (B, Num_Graph_Nodes, FeatureDim)
                                    "F_graph,j"
            correspondence_map_y (torch.Tensor): Binary indicator y_ij.
                                                 Shape (B, Num_Unet_Items, Num_Graph_Nodes).
                                                 y_ij=1 if U-Net item i and graph node j are same region.
            regions_unet (torch.Tensor, optional): Segmentation map from U-Net (B, H, W) or (B, 1, H, W).
            regions_graph (torch.Tensor, optional): Partition map from graph (B, H, W) or (B, 1, H, W).
                                                    (e.g., from min-cut on patch graph).
        
        The paper mentions "pixel (i,j)" in the final segmentation map S (Eq 3),
        but L_feature equation refers to "F_U-Net,i" and "F_graph,j".
        This suggests "i" and "j" are indices of feature vectors, not pixel coords.
        "y_ij is a binary indicator (1 if i and j correspond to the same region, 0 otherwise)"
        This is the most complex part: defining "i", "j", and "y_ij".

        Assumption:
        - f_unet: (B, N_patches_unet, D_feat) - U-Net features per patch.
        - f_graph: (B, N_patches_graph, D_feat) - Graph node (patch) features.
        - correspondence_map_y: (B, N_patches_unet, N_patches_graph)
          If patches are the same for U-Net and graph, N_patches_unet = N_patches_graph = N_patches,
          and y_ij is 1 if i=j and they belong to the same *semantic* region, or simply if i=j.
          The paper's y_ij is "1 if i and j correspond to the same region". This implies a semantic check.

        Let's assume for now that i and j iterate over patches, and y_ij means
        patch_i and patch_j belong to the same *predicted semantic region*.
        If N_unet_items == N_graph_nodes (e.g. both are patches):
        Then y_ij could be 1 if patch_i and patch_j belong to the same semantic class
        AND i==j (comparing a U-Net patch feature to its corresponding graph patch feature).
        Or, it could be a more general contrastive loss: pull features of same-region patches together,
        push apart features of different-region patches.

        The wording "1 if i and j correspond to the same region" is key.
        Let's simplify: Assume f_unet and f_graph are features for the *same set of N patches*.
        So, f_unet is (B, N_patches, D), f_graph is (B, N_patches, D).
        Then i and j iterate over these N_patches.
        y_ij = 1 if patch i and patch j are in the same semantic region (e.g. both 'mango').
        This would make it a contrastive-style loss within f_unet against f_graph.

        The equation is Σ_i Σ_j. If i and j are patch indices:
        L_feature = Σ_p1 Σ_p2 [y_p1,p2 * dist(F_U(p1), F_G(p2))^2 +
                               (1-y_p1,p2) * hinge_loss(m - dist(F_U(p1), F_G(p2)))^2]

        Alternative, more direct interpretation based on "F_U-Net,i - F_graph,j":
        It seems to compare a feature F_U-Net,i (e.g. for pixel/patch i from U-Net output)
        with a feature F_graph,j (e.g. for graph node j).
        If these are features for the *same spatial patch*, then i=j.
        And y_ij = y_i would be 1 if patch i is 'mango', 0 if 'background'.
        This interpretation aligns with the provided equation having only one norm ||.||.

        Let's take F_U-Net,i as features for patch i from U-Net.
        Let's take F_graph,j as features for patch j from the graph (GAT output on patch graph).
        If we assume i and j index the *same set of patches*, then we are comparing F_U-Net,p with F_graph,p for each patch p.
        Then y_p (replacing y_ij) is 1 if patch p is in a 'positive' region, 0 otherwise.
        
        L_feature = Σ_p [y_p * ||F_U(p) - F_G(p)||^2 + (1-y_p) * max(0, m - ||F_U(p) - F_G(p)||)^2]
        Here, y_p indicates if patch p belongs to the target class.
        This seems more plausible. `regions_unet` would provide `y_p`.
        
        Args revised:
            f_unet (torch.Tensor): (B, N_patches, D_feat) - U-Net features per patch.
            f_graph (torch.Tensor): (B, N_patches, D_feat) - Graph (GAT) features per patch.
            patch_region_labels_y (torch.Tensor): (B, N_patches) - Binary label, 1 if patch is target region.
        """
        B, N_patches, D_feat_unet = f_unet.shape
        _, _, D_feat_graph = f_graph.shape

        if f_unet.shape != f_graph.shape:
            # Need to make feature dimensions compatible if they differ.
            # This would require an adapter (e.g., linear layer).
            # For now, assume D_feat_unet == D_feat_graph for simplicity.
            # Or, only compare if they are indeed different views of the same patch.
             raise ValueError(f"f_unet ({f_unet.shape}) and f_graph ({f_graph.shape}) must have same dimensions for this loss version.")

        if correspondence_map_y.shape != (B, N_patches): # Assuming patch_region_labels_y
             raise ValueError(f"correspondence_map_y (patch_region_labels_y) shape ({correspondence_map_y.shape}) "
                             f"is not (Batch, Num_Patches) = ({B}, {N_patches}).")

        # Ensure y is float for multiplication
        y_p = correspondence_map_y.float().unsqueeze(-1) # (B, N_patches, 1)

        # Euclidean distance squared: ||a - b||^2
        dist_sq = torch.sum((f_unet - f_graph)**2, dim=2) # (B, N_patches)

        # Positive term: y_p * dist_sq
        loss_positive = y_p.squeeze(-1) * dist_sq # (B, N_patches)

        # Negative term: (1 - y_p) * max(0, m - dist)^2
        # Note: paper has max(0, m - ||F_U - F_G||^2)^2.
        # If it's max(0, m - ||F_U - F_G|| )^2, then sqrt is needed.
        # Let's use dist = ||F_U - F_G||.
        dist = torch.sqrt(dist_sq + 1e-8) # Add epsilon for stability if dist_sq can be 0
        
        hinge_term = F.relu(self.margin - dist) # max(0, m - dist)
        loss_negative = (1 - y_p.squeeze(-1)) * (hinge_term**2) # (B, N_patches)
        
        total_loss_per_patch = loss_positive + loss_negative # (B, N_patches)
        
        # Sum over patches, then average over batch
        loss = torch.sum(total_loss_per_patch, dim=1).mean()
        
        return loss

if __name__ == '__main__':
    loss_fn = FeatureConsistencyLoss(margin=1.0)
    B, N, D = 2, 10, 32 # Batch, Num_patches, Feature_dim

    f_unet_patches = torch.randn(B, N, D)
    f_graph_patches = torch.randn(B, N, D)
    
    # y_p: 1 if patch is 'mango', 0 if 'background'
    # Let's say first half are mango, second half background
    patch_is_mango_labels = torch.zeros(B, N, dtype=torch.long)
    patch_is_mango_labels[:, :N//2] = 1 

    # Make features for 'mango' patches closer, 'background' further apart
    # For mango patches (y_p=1), we want f_unet and f_graph to be similar.
    f_unet_patches[:, :N//2, :] = f_graph_patches[:, :N//2, :] + torch.rand_like(f_graph_patches[:, :N//2, :]) * 0.1 # Close
    # For background patches (y_p=0), we want them to be far if dist < margin.
    f_unet_patches[:, N//2:, :] = f_graph_patches[:, N//2:, :] + torch.rand_like(f_graph_patches[:, N//2:, :]) * 2.0 # Far


    loss = loss_fn(f_unet_patches, f_graph_patches, patch_is_mango_labels)
    print(f"Feature Consistency Loss: {loss.item()}")

    # Test when all are close, y_p=1 (positive pairs)
    f_graph_close = f_unet_patches + torch.rand_like(f_unet_patches) * 0.01
    y_all_positive = torch.ones(B, N, dtype=torch.long)
    loss_all_pos = loss_fn(f_unet_patches, f_graph_close, y_all_positive)
    print(f"Loss (all positive, close features): {loss_all_pos.item()}") # Should be small

    # Test when all are far, y_p=0 (negative pairs)
    f_graph_far = f_unet_patches + torch.rand_like(f_unet_patches) * 5.0 # Distance > margin
    y_all_negative = torch.zeros(B, N, dtype=torch.long)
    loss_all_neg_far = loss_fn(f_unet_patches, f_graph_far, y_all_negative)
    print(f"Loss (all negative, far features, dist > margin): {loss_all_neg_far.item()}") # Hinge should be 0

    f_graph_close_neg = f_unet_patches + torch.rand_like(f_unet_patches) * 0.1 # Distance < margin
    loss_all_neg_close = loss_fn(f_unet_patches, f_graph_close_neg, y_all_negative)
    print(f"Loss (all negative, close features, dist < margin): {loss_all_neg_close.item()}") # Hinge should be active