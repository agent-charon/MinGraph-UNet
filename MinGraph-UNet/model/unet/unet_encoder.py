import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.relu(x)
        return x

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, init_features=32, depth=4):
        """
        U-Net Encoder (Downsampling Path).

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            init_features (int): Number of features in the first convolutional layer.
                                 This will be doubled at each depth level.
            depth (int): Number of downsampling blocks (pools).
        """
        super().__init__()
        self.depth = depth
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        features = init_features
        current_in_channels = in_channels

        for i in range(depth):
            self.encoder_blocks.append(ConvBlock(current_in_channels, features))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_in_channels = features
            features *= 2 # Double features for the next block

        # Bottleneck layer (deepest part of the U)
        self.bottleneck = ConvBlock(current_in_channels, features) # features is already doubled from last loop iter

    def forward(self, x):
        """
        Forward pass through the encoder.

        Returns:
            list: A list of feature maps from each encoder block (skip connections)
                  before pooling, ordered from shallowest to deepest.
            torch.Tensor: The feature map from the bottleneck layer.
        """
        skip_connections = []
        current_x = x

        for i in range(self.depth):
            current_x = self.encoder_blocks[i](current_x)
            skip_connections.append(current_x)
            current_x = self.pool_layers[i](current_x)
        
        bottleneck_out = self.bottleneck(current_x)
        
        return skip_connections, bottleneck_out

if __name__ == '__main__':
    # Example Usage
    dummy_input = torch.randn(2, 3, 128, 128) # Batch_size=2, C=3, H=128, W=128
    
    # Test with default parameters
    encoder = UNetEncoder(in_channels=3, init_features=32, depth=4)
    skip_outputs, bottleneck_output = encoder(dummy_input)

    print("Encoder Output Shapes (default depth=4, init_features=32):")
    for i, skip in enumerate(skip_outputs):
        print(f"  Skip connection {i}: {skip.shape}")
    print(f"  Bottleneck output: {bottleneck_output.shape}")
    # Expected for depth=4, init_feat=32, input 128x128:
    # Skip 0: (B, 32, 128, 128) -> Pool -> (B, 32, 64, 64)
    # Skip 1: (B, 64, 64, 64)   -> Pool -> (B, 64, 32, 32)
    # Skip 2: (B, 128, 32, 32)  -> Pool -> (B, 128, 16, 16)
    # Skip 3: (B, 256, 16, 16)  -> Pool -> (B, 256, 8, 8)
    # Bottleneck: (B, 512, 8, 8)

    # Test with different depth
    encoder_d2 = UNetEncoder(in_channels=3, init_features=64, depth=2)
    skip_outputs_d2, bottleneck_output_d2 = encoder_d2(dummy_input)
    print("\nEncoder Output Shapes (depth=2, init_features=64):")
    for i, skip in enumerate(skip_outputs_d2):
        print(f"  Skip connection {i}: {skip.shape}")
    print(f"  Bottleneck output: {bottleneck_output_d2.shape}")
    # Expected for depth=2, init_feat=64, input 128x128:
    # Skip 0: (B, 64, 128, 128) -> Pool -> (B, 64, 64, 64)
    # Skip 1: (B, 128, 64, 64)  -> Pool -> (B, 128, 32, 32)
    # Bottleneck: (B, 256, 32, 32)