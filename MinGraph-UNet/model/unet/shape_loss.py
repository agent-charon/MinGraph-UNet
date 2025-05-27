import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EllipticalShapeLoss(nn.Module):
    """
    Enforces elliptical shape constraints using Mahalanobis distance.
    L_shape = Σ_i Σ_j (p_j^T Σ_i^-1 p_j - 1)^2
    where p_j is a pixel coordinate relative to the i-th object's centroid,
    and Σ_i is the covariance matrix of the i-th object's pixels.
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon # For numerical stability with inverse

    def forward(self, segmentation_probs, object_masks_list=None):
        """
        Calculates the elliptical shape loss.

        Args:
            segmentation_probs (torch.Tensor): Softmax probabilities for each class (B, C, H, W).
                                               We'll derive object instances from this.
            object_masks_list (list of list of torch.Tensor, optional):
                If provided, a list (batch) of lists (objects in image) of binary masks (H, W) for each object.
                If None, objects are derived from segmentation_probs (e.g., for the foreground class).
                This is complex as it requires instance segmentation first.
                The paper formulation implies per-object calculation.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # This loss is non-trivial to implement directly from segmentation_probs
        # without first performing instance segmentation to identify individual objects.
        # The paper (Eq. after (3)) refers to "i-th object".

        # Simplification: Apply to the primary foreground class if only semantic segmentation is available.
        # Assume class 1 is the foreground 'mango'.
        # This would mean treating the entire foreground blob as one "object" for shape regularization,
        # which might not be the intention if multiple mangoes are present.

        # A more robust implementation would require:
        # 1. Instance segmentation (e.g., from a separate model or connected components on seg_map).
        # 2. For each instance:
        #    a. Get pixel coordinates (p_j).
        #    b. Calculate centroid.
        #    c. Calculate covariance matrix (Σ_i).
        #    d. Compute Mahalanobis distance and the loss term.

        # --- Simplified Placeholder / Conceptual ---
        # Let's assume we have a way to get binary masks for each predicted object instance.
        # If `object_masks_list` is not provided, this loss might be hard to compute meaningfully
        # directly from semantic segmentation probabilities without instance proposals.

        # The paper's formulation: "where p_j are the coordinates of the j-th pixel in the i-th object,
        # centered at the ellipse's centroid, and Σi is the covariance matrix of the ellipse for the i-th object."
        # This means Σi is derived from the object's pixels, not a predefined ellipse.

        if object_masks_list is None:
            # Attempt to derive objects from segmentation_probs (highly simplified)
            # This is a placeholder and needs a proper instance segmentation step.
            B, C, H, W = segmentation_probs.shape
            if C <= 1: # No foreground class to analyze
                return torch.tensor(0.0, device=segmentation_probs.device)

            # Consider the class with highest probability (excluding background if class 0)
            # Or a specific foreground class index.
            foreground_class_idx = 1 # Assume mango is class 1
            if C <= foreground_class_idx:
                 return torch.tensor(0.0, device=segmentation_probs.device)

            # Create binary masks for the foreground class per batch item
            # This is still semantic, not instance.
            pred_labels = torch.argmax(segmentation_probs, dim=1) # (B, H, W)
            
            # This is where true instance segmentation would be needed.
            # For now, let's operate on the whole foreground region if this path is taken.
            # This simplification might not match the paper's intent well.
            # print("Warning: EllipticalShapeLoss is using a simplified object derivation (semantic foreground).")
            # print("         A proper instance segmentation approach is recommended for this loss.")
            
            batch_loss = 0.0
            num_objects_processed = 0
            for b in range(B):
                # Simple connected components can give "instances"
                # from skimage.measure import label
                # binary_mask_np = (pred_labels[b] == foreground_class_idx).cpu().numpy().astype(np.uint8)
                # labeled_mask, num_labels = label(binary_mask_np, connectivity=2, background=0, return_num=True)
                # current_object_masks = []
                # for i in range(1, num_labels + 1):
                #     current_object_masks.append(torch.from_numpy(labeled_mask == i).to(segmentation_probs.device))
                # If using this path, the loop below should iterate `current_object_masks`.
                # For now, using a single foreground blob as one object:
                single_object_mask = (pred_labels[b] == foreground_class_idx)
                if single_object_mask.sum() < 10: # Skip tiny "objects"
                    continue
                object_masks_for_batch_item = [single_object_mask] # Treat whole foreground as one object

                for object_mask in object_masks_for_batch_item: # Iterate over derived "objects"
                    if object_mask.sum() < 10: # Min number of pixels for stable covariance
                        continue
                    
                    # Get pixel coordinates (y, x) of the object
                    coords = torch.nonzero(object_mask, as_tuple=False).float() # (Num_pixels, 2) -> (row, col)
                    
                    if coords.shape[0] < 2: # Need at least 2 points for covariance
                        continue

                    # Calculate centroid
                    centroid = torch.mean(coords, dim=0) # (2,) -> (mean_row, mean_col)

                    # Center coordinates
                    centered_coords = coords - centroid # (Num_pixels, 2)

                    # Calculate covariance matrix
                    # Σ = (1/N) * X^T X  where X is centered_coords
                    # PyTorch doesn't have a direct cov matrix for 2D points like np.cov.
                    # We need (X-mu)^T (X-mu) / N
                    # cov_matrix shape should be (2, 2)
                    # For (y,x) coords: cov = [[var(y), cov(y,x)], [cov(x,y), var(x)]]
                    if centered_coords.shape[0] <= 1: continue # Need more than 1 point

                    # Manual covariance:
                    # centered_coords is (N, 2). N points, 2 dimensions (y, x)
                    # cov_matrix = (centered_coords.T @ centered_coords) / (centered_coords.shape[0] -1) # Correct for sample cov
                    # However, torch.cov is available in newer PyTorch versions.
                    try:
                        # torch.cov expects features in rows, observations in columns (or vice-versa with rowvar)
                        # centered_coords.T is (2, N_pixels)
                        cov_matrix = torch.cov(centered_coords.T) # Default is rowvar=True
                    except RuntimeError as e: # e.g. if too few points
                        # print(f"Skipping object due to covariance error: {e}")
                        continue
                    
                    # Add epsilon for numerical stability for inverse
                    cov_matrix_inv = torch.inverse(cov_matrix + self.epsilon * torch.eye(2, device=cov_matrix.device))

                    # Calculate Mahalanobis distance term p_j^T Σ_i^-1 p_j for each pixel
                    # p_j is a row in centered_coords (1, 2)
                    # (p_j @ cov_inv) @ p_j.T
                    # Efficiently: diag( centered_coords @ cov_inv @ centered_coords.T )
                    mahalanobis_terms = torch.diag(centered_coords @ cov_matrix_inv @ centered_coords.T)
                    
                    loss_per_object = torch.mean((mahalanobis_terms - 1.0)**2)
                    batch_loss += loss_per_object
                    num_objects_processed +=1
            
            return batch_loss / num_objects_processed if num_objects_processed > 0 else torch.tensor(0.0, device=segmentation_probs.device)

        else: # object_masks_list is provided
            # This is the more intended path for this loss.
            # `object_masks_list` should be: B x [list of object_mask (H,W) tensors]
            batch_loss = 0.0
            num_objects_processed = 0
            for b_idx in range(len(object_masks_list)):
                masks_in_image = object_masks_list[b_idx]
                for object_mask in masks_in_image: # object_mask is (H,W) boolean/binary
                    if object_mask.sum() < 10: # Min number of pixels
                        continue
                    
                    coords = torch.nonzero(object_mask, as_tuple=False).float()
                    if coords.shape[0] < 2: continue

                    centroid = torch.mean(coords, dim=0)
                    centered_coords = coords - centroid
                    if centered_coords.shape[0] <= 1: continue
                    
                    try:
                        cov_matrix = torch.cov(centered_coords.T)
                    except RuntimeError:
                        continue
                    
                    cov_matrix_inv = torch.inverse(cov_matrix + self.epsilon * torch.eye(2, device=cov_matrix.device))
                    mahalanobis_terms = torch.diag(centered_coords @ cov_matrix_inv @ centered_coords.T)
                    
                    loss_per_object = torch.mean((mahalanobis_terms - 1.0)**2)
                    batch_loss += loss_per_object
                    num_objects_processed += 1
            
            return batch_loss / num_objects_processed if num_objects_processed > 0 else torch.tensor(0.0, device=segmentation_probs.device)


if __name__ == '__main__':
    loss_fn = EllipticalShapeLoss()

    # --- Test Case 1: Ideal Elliptical Object Mask ---
    H, W = 64, 64
    obj_mask1 = torch.zeros(H, W, dtype=torch.bool)
    center_y, center_x = H // 2, W // 2
    radius_y, radius_x = H // 4, W // 3 # Ellipse radii
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    ellipse_eq = ((y_coords - center_y) / radius_y)**2 + ((x_coords - center_x) / radius_x)**2
    obj_mask1[ellipse_eq <= 1] = True
    
    # Create dummy segmentation_probs (not really used if object_masks_list is provided)
    dummy_seg_probs = torch.rand(1, 2, H, W) 
    
    object_masks = [[obj_mask1.cuda()]] if torch.cuda.is_available() else [[obj_mask1]]
    loss_val1 = loss_fn(dummy_seg_probs.cuda() if torch.cuda.is_available() else dummy_seg_probs, object_masks)
    print(f"Loss for ideal ellipse mask: {loss_val1.item()}") # Should be close to 0

    # --- Test Case 2: Non-Elliptical Object Mask (e.g., a square) ---
    obj_mask2 = torch.zeros(H, W, dtype=torch.bool)
    obj_mask2[H//4:3*H//4, W//4:3*W//4] = True # A square
    object_masks2 = [[obj_mask2.cuda()]] if torch.cuda.is_available() else [[obj_mask2]]
    loss_val2 = loss_fn(dummy_seg_probs.cuda() if torch.cuda.is_available() else dummy_seg_probs, object_masks2)
    print(f"Loss for square mask: {loss_val2.item()}") # Should be > loss_val1

    # --- Test Case 3: Using segmentation_probs (simplified path, needs skimage) ---
    print("\nTesting simplified path (deriving objects from seg_probs):")
    # Create seg_probs where class 1 forms an ellipse
    seg_probs_test = torch.zeros(1, 2, H, W) # B, C, H, W
    # Background (class 0) has higher prob everywhere initially
    seg_probs_test[:, 0, :, :] = 0.7
    seg_probs_test[:, 1, :, :] = 0.3
    # Make class 1 (foreground) higher prob within the ellipse
    seg_probs_test[0, 1, obj_mask1] = 0.8 
    seg_probs_test[0, 0, obj_mask1] = 0.2

    # This path might need skimage.measure.label if not available, it will be very basic
    try:
        from skimage.measure import label # Test if available
        loss_val3 = loss_fn(seg_probs_test.cuda() if torch.cuda.is_available() else seg_probs_test, object_masks_list=None)
        print(f"Loss from seg_probs (ellipse like): {loss_val3.item()}")
    except ImportError:
        print("skimage.measure.label not found, simplified path using single foreground blob will run.")
        loss_val3_simple_blob = loss_fn(seg_probs_test.cuda() if torch.cuda.is_available() else seg_probs_test, object_masks_list=None)
        print(f"Loss from seg_probs (ellipse like, single blob): {loss_val3_simple_blob.item()}")

    # Make class 1 form a square
    seg_probs_test_sq = torch.zeros(1, 2, H, W)
    seg_probs_test_sq[:, 0, :, :] = 0.7
    seg_probs_test_sq[:, 1, :, :] = 0.3
    seg_probs_test_sq[0, 1, obj_mask2] = 0.8 
    seg_probs_test_sq[0, 0, obj_mask2] = 0.2
    try:
        from skimage.measure import label
        loss_val4 = loss_fn(seg_probs_test_sq.cuda() if torch.cuda.is_available() else seg_probs_test_sq, object_masks_list=None)
        print(f"Loss from seg_probs (square like): {loss_val4.item()}")
    except ImportError:
        loss_val4_simple_blob = loss_fn(seg_probs_test_sq.cuda() if torch.cuda.is_available() else seg_probs_test_sq, object_masks_list=None)
        print(f"Loss from seg_probs (square like, single blob): {loss_val4_simple_blob.item()}")

    # Note: The EllipticalShapeLoss is sensitive and requires careful instance segmentation.
    # The simplified path (object_masks_list=None) is a rough approximation.