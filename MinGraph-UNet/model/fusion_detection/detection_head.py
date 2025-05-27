import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_features_channels, num_classes, num_detection_outputs=5, 
                 fc_hidden_dim=256, input_is_flat=False):
        """
        A simple detection head that takes fused features and predicts bounding boxes and scores.
        "The fused features are passed through convolutional layers and a fully connected network
         to predict bounding boxes b and confidence scores s" (Sec 3.6)

        Args:
            in_features_channels (int): Number of channels in the input fused feature map F_f (C_fused).
                                        Assumes F_f is (B, C_fused, H_f, W_f).
            num_classes (int): Number of object classes (e.g., 1 for 'mango' if background is implicit).
                               This is for class scores if predicting per class.
            num_detection_outputs (int): Number of outputs per detected object.
                                         Typically 5 for (confidence, x, y, w, h).
                                         Or 4 for (x,y,w,h) + num_classes for class scores + 1 for objectness.
            fc_hidden_dim (int): Dimension for hidden FC layers.
            input_is_flat (bool): If True, input `f_fused` is already (B, FlatFeatureDim).
                                 If False, `f_fused` is (B, C, H, W) and needs pooling/flattening.
        """
        super().__init__()
        self.input_is_flat = input_is_flat
        self.num_classes = num_classes # If predicting class scores separately

        # Convolutional layers (if input is not flat)
        # These are examples; the paper doesn't specify exact architecture.
        # These convs can further reduce spatial dimensions and refine features.
        if not self.input_is_flat:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_features_channels, in_features_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(in_features_channels//2),
                nn.Conv2d(in_features_channels // 2, in_features_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(in_features_channels//4),
                nn.AdaptiveAvgPool2d((1, 1)) # Pool to (B, C_final_conv, 1, 1)
            )
            fc_input_dim = in_features_channels // 4
        else:
            fc_input_dim = in_features_channels # If input is already flat

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Output layers
        # Bounding box regression (x, y, w, h) - 4 values
        self.fc_bbox = nn.Linear(fc_hidden_dim // 2, 4)
        
        # Confidence score (objectness) - 1 value
        self.fc_confidence = nn.Linear(fc_hidden_dim // 2, 1)

        # Class scores (if multi-class detection) - num_classes values
        # If num_classes=1 (e.g. just 'mango'), this might be combined with confidence.
        # The paper mentions "s, b = FFN(F_f)". 's' could be confidence or class scores.
        # If it's multi-class, 's' would include class probabilities.
        if self.num_classes > 1: # For multi-class detection problems
            self.fc_class_scores = nn.Linear(fc_hidden_dim // 2, self.num_classes)


    def forward(self, f_fused):
        """
        Args:
            f_fused (torch.Tensor): Fused features.
                                    If not flat: (B, C_fused, H_f, W_f).
                                    If flat: (B, FlatFeatureDim).

        Returns:
            torch.Tensor: Bounding box predictions (B, NumAnchors_or_Regions, 4). (x,y,w,h), normalized 0-1 or pixel.
            torch.Tensor: Confidence scores (B, NumAnchors_or_Regions, 1). Sigmoid applied.
            torch.Tensor (optional): Class scores (B, NumAnchors_or_Regions, NumClasses). Softmax/Sigmoid per class.
            
            Note: This simple head predicts one box per input feature set (after pooling).
            For multiple detections, an anchor-based approach (like SSD/YOLO heads) or
            region proposal network (like Faster R-CNN) would be needed.
            The paper implies a direct FFN(F_f) -> s, b. This suggests one set of s,b per input sample's F_f.
            This is more like image-level classification/regression unless F_f is processed per region.
            
            Let's assume this head operates on globally pooled features for now, predicting one primary box.
            This is a simplification. A real detection head would be more complex.
        """
        if not self.input_is_flat:
            x = self.conv_block(f_fused) # (B, C_final_conv, 1, 1)
            x = torch.flatten(x, 1)      # (B, C_final_conv)
        else:
            x = f_fused                  # (B, FlatFeatureDim)

        x = self.fc_layers(x) # (B, fc_hidden_dim // 2)

        # Bbox predictions (x,y,w,h) - typically normalized 0-1 or direct pixel values
        # Using sigmoid for normalized coordinates/sizes (0-1 range)
        bboxes = torch.sigmoid(self.fc_bbox(x)) # (B, 4)

        # Confidence score (objectness)
        confidence = torch.sigmoid(self.fc_confidence(x)) # (B, 1)
        
        # Class scores
        if self.num_classes > 1:
            class_scores = self.fc_class_scores(x) # (B, num_classes)
            # Apply softmax if classes are mutually exclusive, or sigmoid if multi-label per box
            # class_scores = F.softmax(class_scores, dim=-1) # For single class per box
            return bboxes, confidence, class_scores
        else:
            # For binary (object vs background), confidence score might be enough
            return bboxes, confidence


if __name__ == '__main__':
    B = 4
    C_fused_test = 256
    H_f, W_f = 16, 16 # Spatial dim of fused features
    num_cls_test = 1 # e.g., just "mango" vs background (confidence score handles this)
    
    # --- Test 1: Input is a feature map ---
    print("--- Test 1: Input is feature map (B, C, H, W) ---")
    f_fused_map = torch.randn(B, C_fused_test, H_f, W_f)
    det_head1 = DetectionHead(in_features_channels=C_fused_test, num_classes=num_cls_test, input_is_flat=False)
    
    if num_cls_test > 1:
        b_out1, c_out1, cls_out1 = det_head1(f_fused_map)
        print(f"BBoxes shape: {b_out1.shape}") # (B, 4)
        print(f"Confidence shape: {c_out1.shape}") # (B, 1)
        print(f"Class scores shape: {cls_out1.shape}")# (B, num_cls_test)
    else:
        b_out1, c_out1 = det_head1(f_fused_map)
        print(f"BBoxes shape: {b_out1.shape}") # (B, 4)
        print(f"Confidence shape: {c_out1.shape}") # (B, 1)

    # --- Test 2: Input is already flat ---
    print("\n--- Test 2: Input is flat feature vector (B, D) ---")
    flat_feature_dim = 512
    f_fused_flat = torch.randn(B, flat_feature_dim)
    det_head2 = DetectionHead(in_features_channels=flat_feature_dim, num_classes=num_cls_test, input_is_flat=True)
    
    if num_cls_test > 1:
        b_out2, c_out2, cls_out2 = det_head2(f_fused_flat)
        print(f"BBoxes shape: {b_out2.shape}")
        print(f"Confidence shape: {c_out2.shape}")
        print(f"Class scores shape: {cls_out2.shape}")
    else:
        b_out2, c_out2 = det_head2(f_fused_flat)
        print(f"BBoxes shape: {b_out2.shape}")
        print(f"Confidence shape: {c_out2.shape}")

    # --- Test 3: Multi-class detection ---
    num_cls_multi = 3
    print(f"\n--- Test 3: Multi-class (num_classes={num_cls_multi}) ---")
    det_head3 = DetectionHead(in_features_channels=C_fused_test, num_classes=num_cls_multi, input_is_flat=False)
    b_out3, c_out3, cls_out3 = det_head3(f_fused_map)
    print(f"BBoxes shape: {b_out3.shape}")
    print(f"Confidence shape: {c_out3.shape}")
    print(f"Class scores shape: {cls_out3.shape}") # (B, num_cls_multi)