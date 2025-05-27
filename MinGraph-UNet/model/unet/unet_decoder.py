import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_encoder import ConvBlock # Reusing ConvBlock

class DecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_prev, out_channels, use_batchnorm=True):
        """
        A single block in the U-Net Decoder (Upsampling Path).
        It takes features from the previous decoder layer and skip connection.

        Args:
            in_channels_skip (int): Number of channels from the skip connection (encoder).
            in_channels_prev (int): Number of channels from the previous (deeper) decoder block.
            out_channels (int): Number of output channels for this block.
        """
        super().__init__()
        # Upsample the features from the previous decoder block
        # The total input channels to the ConvBlock will be in_channels_skip + in_channels_prev / 2 (if upsampling halves channels)
        # Or in_channels_skip + upsampled_prev_channels
        
        # We upsample prev_features to match skip_features spatial dim, then concat.
        # The number of channels for the upsampled features will be in_channels_prev.
        # So, ConvBlock input channels = in_channels_skip + in_channels_prev.
        self.upsample = nn.ConvTranspose2d(in_channels_prev, in_channels_prev // 2, kernel_size=2, stride=2)
        # After upsampling and concat, the conv block handles the channel reduction.
        # Input to conv_block is (in_channels_skip + in_channels_prev // 2)
        self.conv_block = ConvBlock(in_channels_skip + (in_channels_prev // 2) , out_channels, use_batchnorm=use_batchnorm)

    def forward(self, x_prev, x_skip):
        """
        Args:
            x_prev (torch.Tensor): Features from the previous (deeper) decoder block.
            x_skip (torch.Tensor): Features from the corresponding encoder skip connection.
        """
        x_up = self.upsample(x_prev)
        
        # Handle potential size mismatch due to pooling/convs (common in U-Net)
        # Pad x_up if its spatial dimensions are smaller than x_skip
        # Or crop x_skip if its spatial dimensions are larger than x_up
        # PyTorch's ConvTranspose2d with kernel=2, stride=2 usually doubles spatial size.
        # If skip connection size is odd, there might be a 1-pixel difference.
        
        diff_y = x_skip.size()[2] - x_up.size()[2]
        diff_x = x_skip.size()[3] - x_up.size()[3]

        # Pad x_up to match x_skip
        # (padding_left, padding_right, padding_top, padding_bottom)
        x_up = F.pad(x_up, [diff_x // 2, diff_x - diff_x // 2,
                           diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along the channel dimension
        x_concat = torch.cat([x_skip, x_up], dim=1)
        
        x_out = self.conv_block(x_concat)
        return x_out

class UNetDecoder(nn.Module):
    def __init__(self, num_classes, init_features=32, depth=4):
        """
        U-Net Decoder (Upsampling Path).

        Args:
            num_classes (int): Number of output classes for segmentation.
            init_features (int): Number of features in the first encoder block.
                                 This helps determine the channel counts for skip connections.
            depth (int): Number of upsampling blocks, should match encoder depth.
        """
        super().__init__()
        self.depth = depth
        self.decoder_blocks = nn.ModuleList()

        # Calculate feature channels at each level of the encoder/decoder
        # features_levels[0] = init_features, features_levels[1] = init_features*2, etc.
        # Bottleneck features will be init_features * (2**depth)
        # Skip connection features will be init_features * (2**i) for i-th skip
        
        # Channels for skip connections (from encoder, shallow to deep):
        # init_features, init_features*2, init_features*4, ...
        # Channels for previous decoder block outputs (from bottleneck up to shallow):
        # Bottleneck: init_features * (2**depth)
        # Dec Block 1 (deepest): takes bottleneck (prev) and skip_conn[depth-1]
        #   - prev_in_ch = init_features * (2**depth)
        #   - skip_in_ch = init_features * (2**(depth-1))
        #   - out_ch = init_features * (2**(depth-1))
        # Dec Block 2: takes output of Dec Block 1 (prev) and skip_conn[depth-2]
        #   - prev_in_ch = init_features * (2**(depth-1))
        #   - skip_in_ch = init_features * (2**(depth-2))
        #   - out_ch = init_features * (2**(depth-2))
        # ...and so on.

        # The features list for the decoder should go from deep to shallow
        # (i.e., from bottleneck size upwards)
        
        # Number of features in the bottleneck layer (output of encoder)
        # Example: init_features=32, depth=4. Encoder features: 32, 64, 128, 256. Bottleneck: 512.
        # Skip connections (from shallow to deep): 32, 64, 128, 256
        # Decoder takes bottleneck (512) and skip (256) -> output (256)
        # Takes dec_out (256) and skip (128) -> output (128)
        # Takes dec_out (128) and skip (64)  -> output (64)
        # Takes dec_out (64)  and skip (32)  -> output (32)

        prev_block_channels = init_features * (2**depth) # From bottleneck

        for i in reversed(range(depth)): # from depth-1 down to 0
            skip_conn_channels = init_features * (2**i)
            decoder_out_channels = init_features * (2**i) # Output of this decoder block
            
            self.decoder_blocks.append(
                DecoderBlock(in_channels_skip=skip_conn_channels,
                             in_channels_prev=prev_block_channels,
                             out_channels=decoder_out_channels)
            )
            prev_block_channels = decoder_out_channels # Output of current becomes input for next (shallower)

        # Final 1x1 convolution to map to num_classes
        self.final_conv = nn.Conv2d(prev_block_channels, num_classes, kernel_size=1)


    def forward(self, skip_connections, bottleneck_out):
        """
        Forward pass through the decoder.

        Args:
            skip_connections (list): List of feature maps from encoder (shallowest to deepest).
            bottleneck_out (torch.Tensor): Feature map from the U-Net bottleneck.

        Returns:
            torch.Tensor: Segmentation logits (B, Num_Classes, H, W).
            list: List of feature maps from each decoder block (deepest to shallowest, before final_conv).
                  These can be used for `F_u` in the paper if multi-scale U-Net features are needed.
        """
        # skip_connections are ordered from shallowest to deepest. We need to reverse for decoder.
        reversed_skips = skip_connections[::-1] 
        
        current_x = bottleneck_out
        decoder_feature_maps = []

        for i in range(self.depth):
            current_x = self.decoder_blocks[i](current_x, reversed_skips[i])
            decoder_feature_maps.append(current_x)
            
        logits = self.final_conv(current_x) # current_x is the output of the shallowest decoder block
        
        # decoder_feature_maps are from deepest to shallowest. Paper might want shallow to deep or specific scales.
        # The paper's F_u are "features Fr from multiple stages of the U-Net" (Sec 3.3) and
        # "U-Net features Fu at multiple scales" (Sec 3.6). This could mean encoder skips,
        # decoder outputs, or a mix. For now, let's return decoder outputs.
        return logits, decoder_feature_maps[::-1] # Return shallowest to deepest for F_u consistency

if __name__ == '__main__':
    # Example Usage (requires UNetEncoder for dummy data)
    from .unet_encoder import UNetEncoder
    dummy_input = torch.randn(2, 3, 128, 128) # B, C, H, W
    
    init_feats = 32
    depth_val = 4
    num_cls = 5 # Example number of classes

    encoder = UNetEncoder(in_channels=3, init_features=init_feats, depth=depth_val)
    skips, bottleneck = encoder(dummy_input)
    
    decoder = UNetDecoder(num_classes=num_cls, init_features=init_feats, depth=depth_val)
    logits_out, f_u_decoder_features = decoder(skips, bottleneck)

    print(f"\nDecoder Output (num_classes={num_cls}, init_features={init_feats}, depth={depth_val}):")
    print(f"  Logits shape: {logits_out.shape}") # Expected: (B, num_cls, H_original, W_original) -> (2, 5, 128, 128)
    print(f"  F_u (decoder features, shallowest to deepest):")
    for i, feat in enumerate(f_u_decoder_features):
        print(f"    Decoder feature {i}: {feat.shape}")
    # Expected shapes for F_u (decoder_features[::-1]) with depth=4, init_features=32
    # Feat 0 (shallowest): (B, 32, 128, 128)
    # Feat 1: (B, 64, 64, 64)
    # Feat 2: (B, 128, 32, 32)
    # Feat 3 (deepest before bottleneck): (B, 256, 16, 16)

    # Check if logits H,W match input H,W (should be the case for standard U-Net)
    assert logits_out.shape[2] == dummy_input.shape[2] and logits_out.shape[3] == dummy_input.shape[3]