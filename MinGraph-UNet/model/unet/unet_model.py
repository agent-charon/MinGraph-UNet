import torch
import torch.nn as nn
from .unet_encoder import UNetEncoder
from .unet_decoder import UNetDecoder

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, init_features=32, depth=4):
        """
        Full U-Net model.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output segmentation classes.
            init_features (int): Number of features in the first encoder layer.
            depth (int): Depth of the U-Net (number of downsampling/upsampling stages).
        """
        super().__init__()
        self.encoder = UNetEncoder(in_channels=in_channels, init_features=init_features, depth=depth)
        self.decoder = UNetDecoder(num_classes=num_classes, init_features=init_features, depth=depth)

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            torch.Tensor: Segmentation logits (B, Num_Classes, H, W).
            list: List of skip connection features from the encoder (shallow to deep).
            list: List of feature maps from each decoder block (shallow to deep, before final_conv).
                  These are suitable for `F_u` as described in the paper.
        """
        skip_connections, bottleneck_out = self.encoder(x)
        logits, f_u_decoder_features = self.decoder(skip_connections, bottleneck_out)
        return logits, skip_connections, f_u_decoder_features

if __name__ == '__main__':
    # Example Usage
    dummy_input = torch.randn(2, 3, 128, 128) # B, C, H, W
    num_cls = 5
    unet_model = UNet(in_channels=3, num_classes=num_cls, init_features=32, depth=4)

    logits, encoder_skips, decoder_features_Fu = unet_model(dummy_input)

    print("Full U-Net Model Output:")
    print(f"  Logits shape: {logits.shape}") # (B, num_cls, H, W)
    print(f"  Number of encoder skip connections: {len(encoder_skips)}")
    for i, skip in enumerate(encoder_skips):
        print(f"    Encoder skip {i} shape: {skip.shape}")
    
    print(f"  Number of decoder features (F_u): {len(decoder_features_Fu)}")
    for i, feat in enumerate(decoder_features_Fu):
        print(f"    Decoder feature (F_u) {i} shape: {feat.shape}") # Shallow to deep

    # Check dimensions
    assert logits.shape[0] == dummy_input.shape[0]
    assert logits.shape[1] == num_cls
    assert logits.shape[2] == dummy_input.shape[2]
    assert logits.shape[3] == dummy_input.shape[3]