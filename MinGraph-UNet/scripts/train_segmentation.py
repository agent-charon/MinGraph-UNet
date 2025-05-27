import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
import os
import argparse
from tqdm import tqdm

# Assuming model and dataset are in the project structure
# Add project root to Python path or use relative imports carefully
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.unet.unet_model import UNet # If path is set up
# Or: from MinGraph_UNet.model.unet.unet_model import UNet
from preprocessing.image_preprocessing.image_preprocess import ImagePreprocessor
# from utils.mango_dataset import MangoDataset # If you created this file

# Define MangoDataset here if not in a separate file for simplicity of this script
# (Copied MangoDataset class from above for self-containment if needed)
# class MangoDataset(...): ... (as defined before)
# class ImagePreprocessor(...): ... (as defined before for self-containment if needed)

def load_config(config_dir, config_name):
    with open(os.path.join(config_dir, config_name), 'r') as f:
        return yaml.safe_load(f)

def dice_loss(pred, target, smooth=1.):
    pred = torch.softmax(pred, dim=1) # Convert logits to probabilities
    # Consider one-hot encoding for target if it's not already
    # Target: (B, H, W), Pred: (B, C, H, W)
    # One-hot encode target to (B, C, H, W)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    
    intersection = (pred * target_one_hot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice.mean() # Average Dice score over classes and batch

def train_unet_segmentation(config_path):
    # Load configurations
    dataset_cfg = load_config(config_path, "dataset.yaml")
    model_cfg = load_config(config_path, "model.yaml")
    preproc_cfg = load_config(config_path, "preprocessing.yaml")
    train_cfg = load_config(config_path, "training.yaml")

    # Setup device
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Preprocessor
    preprocessor = ImagePreprocessor(
        resize_dim=tuple(preproc_cfg['resize_dim']),
        mean=tuple(preproc_cfg['normalization_mean']),
        std=tuple(preproc_cfg['normalization_std']),
        apply_augmentation=True # Enable augmentation for training
    )

    # Initialize Datasets and DataLoaders
    # Define MangoDataset here or import if it's in utils
    # For now, let's assume MangoDataset class is available in this scope
    train_dataset_path = os.path.join(dataset_cfg['data_root'], dataset_cfg['train_dir'])
    train_dataset = MangoDataset(
        image_dir=os.path.join(train_dataset_path, dataset_cfg['image_folder']),
        mask_dir=os.path.join(train_dataset_path, dataset_cfg['mask_folder']),
        preprocessor=preprocessor,
        num_classes=model_cfg['unet']['out_channels']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers']
    )
    
    # (Optional: Validation DataLoader)
    # val_dataset_path = os.path.join(dataset_cfg['data_root'], dataset_cfg['val_dir'])
    # val_dataset = MangoDataset(...)
    # val_loader = DataLoader(...)


    # Initialize Model (U-Net only for this script)
    unet_config = model_cfg['unet']
    model = UNet(
        in_channels=unet_config['in_channels'],
        num_classes=unet_config['out_channels'],
        init_features=unet_config['init_features'],
        depth=unet_config['depth']
    ).to(device)

    # Loss function (e.g., CrossEntropy + Dice)
    criterion_ce = nn.CrossEntropyLoss()
    # criterion_dice = dice_loss # Using custom Dice loss function

    # Optimizer
    if train_cfg['optimizer'].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    elif train_cfg['optimizer'].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=train_cfg['learning_rate'], momentum=train_cfg['sgd_momentum'], weight_decay=train_cfg['weight_decay'])
    else:
        raise ValueError(f"Optimizer {train_cfg['optimizer']} not supported.")

    # LR Scheduler (optional)
    if train_cfg.get('lr_scheduler'):
        if train_cfg['lr_scheduler'].lower() == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['lr_gamma'])
        # Add other schedulers if needed
    else:
        scheduler = None
        
    # Training Loop
    print("Starting U-Net Segmentation Training...")
    for epoch in range(train_cfg['num_epochs']):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        for i, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device) # (B, H, W), long type

            optimizer.zero_grad()

            # U-Net forward pass
            logits, _, _ = model(images) # We only need logits for segmentation loss

            # Calculate loss
            loss_ce_val = criterion_ce(logits, masks)
            loss_dice_val = dice_loss(logits, masks) # Using custom dice loss
            
            # Combine losses (e.g., weighted sum)
            loss = loss_ce_val + loss_dice_val # Simple sum, adjust weights if needed

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1), 'lr': optimizer.param_groups[0]['lr']})

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")

        if scheduler:
            scheduler.step()

        # (Optional: Validation step)
        # model.eval()
        # with torch.no_grad():
        #     val_loss = 0.0
        #     for val_images, val_masks in val_loader:
        #         # ... calculate validation loss ...
        # print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

        # Save checkpoint
        if (epoch + 1) % train_cfg['save_epoch_interval'] == 0:
            checkpoint_dir = train_cfg.get('checkpoint_dir', 'outputs/checkpoints_unet_seg')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"unet_segmentation_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("U-Net Segmentation Training Finished.")
    final_model_path = os.path.join(train_cfg.get('checkpoint_dir', 'outputs/checkpoints_unet_seg'), "unet_segmentation_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final U-Net model saved to {final_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train U-Net for simple segmentation.")
    parser.add_argument('--config_path', type=str, default="configs/", help="Path to the configuration files directory.")
    args = parser.parse_args()

    # Ensure MangoDataset and ImagePreprocessor are accessible
    # For this example, if they are not in standard Python path, you might need:
    # import sys
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(script_dir, '..'))
    # sys.path.insert(0, project_root)
    # from preprocessing.image_preprocessing.image_preprocess import ImagePreprocessor
    # # And define MangoDataset or import from utils.mango_dataset

    # Create dummy config files for testing if they don't exist
    if not os.path.exists(args.config_path): os.makedirs(args.config_path)
    if not os.path.exists(os.path.join(args.config_path, "dataset.yaml")):
        with open(os.path.join(args.config_path, "dataset.yaml"), 'w') as f:
            yaml.dump({
                'data_root': 'temp_mango_dataset', 'train_dir': 'train', 'val_dir': 'val',
                'image_folder': 'images', 'mask_folder': 'masks', 'num_classes': 2
            }, f)
    if not os.path.exists(os.path.join(args.config_path, "model.yaml")):
        with open(os.path.join(args.config_path, "model.yaml"), 'w') as f:
            yaml.dump({
                'unet': {'in_channels': 3, 'out_channels': 2, 'init_features': 16, 'depth': 2} # Smaller for quick test
            }, f)
    if not os.path.exists(os.path.join(args.config_path, "preprocessing.yaml")):
        with open(os.path.join(args.config_path, "preprocessing.yaml"), 'w') as f:
            yaml.dump({
                'resize_dim': [64, 64], # Smaller for quick test
                'normalization_mean': [0.485, 0.456, 0.406],
                'normalization_std': [0.229, 0.224, 0.225]
            }, f)
    if not os.path.exists(os.path.join(args.config_path, "training.yaml")):
        with open(os.path.join(args.config_path, "training.yaml"), 'w') as f:
            yaml.dump({
                'device': 'cpu', 'batch_size': 2, 'num_epochs': 1, # Quick test
                'learning_rate': 0.001, 'optimizer': 'Adam', 'weight_decay': 1e-4,
                'save_epoch_interval': 1, 'checkpoint_dir': 'outputs/checkpoints_unet_seg_test'
            }, f)

    # Setup dummy dataset for testing the script
    base_dir_test = "temp_mango_dataset" # From dataset.yaml
    train_img_dir_test = os.path.join(base_dir_test, "train", "images")
    train_mask_dir_test = os.path.join(base_dir_test, "train", "masks")
    if not os.path.exists(train_img_dir_test): os.makedirs(train_img_dir_test)
    if not os.path.exists(train_mask_dir_test): os.makedirs(train_mask_dir_test)
    for i in range(4): # Enough for batch_size=2
        cv2.imwrite(os.path.join(train_img_dir_test, f"img_{i}.png"), np.random.randint(0,255,(100,100,3), dtype=np.uint8))
        cv2.imwrite(os.path.join(train_mask_dir_test, f"img_{i}.png"), np.random.randint(0,2,(100,100), dtype=np.uint8))


    train_unet_segmentation(args.config_path)

    # Cleanup dummy files and dirs for test
    import shutil
    if os.path.exists(base_dir_test): shutil.rmtree(base_dir_test)
    if os.path.exists("outputs/checkpoints_unet_seg_test"): shutil.rmtree("outputs/checkpoints_unet_seg_test")
    # Remove dummy configs if desired, or keep them for re-runs.