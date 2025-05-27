import cv2
import numpy as np
import torch
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, resize_dim=(128, 128), 
                 mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225),
                 apply_augmentation=False):
        """
        Initializes the ImagePreprocessor.

        Args:
            resize_dim (tuple): Target (height, width) for resizing.
            mean (list/tuple): Mean for normalization.
            std (list/tuple): Standard deviation for normalization.
            apply_augmentation (bool): Whether to apply data augmentation.
        """
        self.resize_dim = resize_dim # H, W
        self.mean = mean
        self.std = std
        self.apply_augmentation = apply_augmentation

        # Basic transforms (always applied)
        self.base_transforms_list = [
            transforms.ToPILImage(), # Convert numpy array (H,W,C) or tensor (C,H,W) to PIL Image
            transforms.Resize(self.resize_dim), # PIL Image resize
            transforms.ToTensor(), # Convert PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            transforms.Normalize(mean=self.mean, std=self.std)
        ]

        # Augmentation transforms (applied if self.apply_augmentation is True)
        if self.apply_augmentation:
            self.augmentation_transforms_list = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Example
                # Add more augmentations as needed
            ]
            # Insert augmentations before ToTensor and Normalize if they expect PIL images
            pil_augmentations = [
                transforms.ToPILImage(),
                transforms.Resize(self.resize_dim), # Ensure size consistency before random crops if any
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                # transforms.RandomResizedCrop(size=self.resize_dim, scale=(0.8, 1.0)), # Example
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
            self.transform = transforms.Compose(pil_augmentations)

        else:
            self.transform = transforms.Compose(self.base_transforms_list)
            

    def preprocess(self, image_path_or_array):
        """
        Loads an image, resizes, normalizes, and optionally augments it.

        Args:
            image_path_or_array (str or np.ndarray): Path to the image or a NumPy array (H, W, C) in BGR format (OpenCV default).

        Returns:
            torch.Tensor: Preprocessed image tensor (C, H, W).
        """
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path_or_array}")
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array
        else:
            raise TypeError("Input must be an image path (str) or a NumPy array.")

        # OpenCV loads images in BGR format. Convert to RGB for PyTorch/PIL.
        if image.ndim == 3 and image.shape[2] == 3: # Color image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 2: # Grayscale image, convert to 3 channels for consistency
             image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image # Assume already in correct format if not BGR

        processed_image_tensor = self.transform(image_rgb)
        return processed_image_tensor

    def preprocess_mask(self, mask_path_or_array, num_classes):
        """
        Loads and preprocesses a segmentation mask.
        Typically involves resizing and converting to a LongTensor with class indices.
        Normalization is usually not applied to masks.

        Args:
            mask_path_or_array (str or np.ndarray): Path to the mask or a NumPy array (H, W).
                                                    Assumes pixel values are class indices.
            num_classes (int): Number of segmentation classes.

        Returns:
            torch.Tensor: Preprocessed mask tensor (H, W) of type Long.
        """
        if isinstance(mask_path_or_array, str):
            # Load as grayscale. For masks, usually 0 for background, 1, 2,... for classes
            mask = cv2.imread(mask_path_or_array, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found at {mask_path_or_array}")
        elif isinstance(mask_path_or_array, np.ndarray):
            mask = mask_path_or_array
            if mask.ndim == 3 and mask.shape[2] == 1: # (H, W, 1)
                mask = mask.squeeze(2)
            elif mask.ndim == 3 and mask.shape[2] > 1: # e.g. one-hot encoded
                 # Assuming class index is in the first channel or needs argmax
                 # This part depends heavily on your mask format
                 print("Warning: Multi-channel mask detected. Assuming class indices are needed.")
                 mask = np.argmax(mask, axis=2) # Example for one-hot
        else:
            raise TypeError("Mask input must be a path (str) or a NumPy array.")

        # Resize the mask to match the image preprocessed dimensions
        # Interpolation method is important for masks: use NEAREST to avoid creating new class values.
        mask_resized = cv2.resize(mask, (self.resize_dim[1], self.resize_dim[0]), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask values are within [0, num_classes-1]
        mask_resized = np.clip(mask_resized, 0, num_classes - 1)
        
        mask_tensor = torch.from_numpy(mask_resized).long() # Convert to LongTensor for CrossEntropyLoss
        return mask_tensor


if __name__ == '__main__':
    # Create a dummy image and mask for testing
    dummy_image_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite("dummy_image.png", dummy_image_bgr)
    
    # For a 2-class problem (background=0, mango=1)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_mask[100:150, 100:150] = 1 # A square of class 1 (mango)
    cv2.imwrite("dummy_mask.png", dummy_mask)

    # Test without augmentation
    preprocessor_no_aug = ImagePreprocessor(resize_dim=(128, 128), apply_augmentation=False)
    img_tensor_no_aug = preprocessor_no_aug.preprocess("dummy_image.png")
    mask_tensor_no_aug = preprocessor_no_aug.preprocess_mask("dummy_mask.png", num_classes=2)
    print("Without Augmentation:")
    print("Image tensor shape:", img_tensor_no_aug.shape) # Expected: torch.Size([3, 128, 128])
    print("Mask tensor shape:", mask_tensor_no_aug.shape)   # Expected: torch.Size([128, 128])
    print("Mask tensor dtype:", mask_tensor_no_aug.dtype) # Expected: torch.int64

    # Test with augmentation
    preprocessor_with_aug = ImagePreprocessor(resize_dim=(128, 128), apply_augmentation=True)
    img_tensor_with_aug = preprocessor_with_aug.preprocess(dummy_image_bgr) # Pass array directly
    # Mask augmentation should ideally be consistent with image augmentation if geometric
    # For simplicity here, we'll just preprocess it without geometric augmentation
    # In a real Dataset, you'd apply synced augmentations to image and mask.
    mask_tensor_aug_style = preprocessor_no_aug.preprocess_mask(dummy_mask, num_classes=2) # Pass array
    print("\nWith Augmentation (Image only for this example):")
    print("Image tensor shape:", img_tensor_with_aug.shape)
    
    # Clean up dummy files
    os.remove("dummy_image.png")
    os.remove("dummy_mask.png")