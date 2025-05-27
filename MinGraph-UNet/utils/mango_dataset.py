# (Can be in a new file like utils/mango_dataset.py or defined within the training script)
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 # For mask loading if needed
import numpy as np

# Assuming image_preprocess.py is accessible
# from MinGraph_UNet.preprocessing.image_preprocessing.image_preprocess import ImagePreprocessor # Adjust path

class MangoDataset(Dataset):
    def __init__(self, image_dir, mask_dir, preprocessor, num_classes, file_extension="*.png"):
        """
        Args:
            image_dir (str): Directory with all input images.
            mask_dir (str): Directory with all corresponding masks.
            preprocessor (ImagePreprocessor instance): For image and mask preprocessing.
            num_classes (int): Number of segmentation classes.
            file_extension (str): Glob pattern for image files (e.g., "*.png", "*.jpg").
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.preprocessor = preprocessor
        self.num_classes = num_classes

        self.image_files = sorted(glob.glob(os.path.join(image_dir, file_extension)))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, file_extension)))

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images found in {image_dir} with pattern {file_extension}")
        if len(self.mask_files) == 0:
             # Allow running without masks for pure inference if mask_dir is None
            if mask_dir is not None:
                print(f"Warning: No masks found in {mask_dir}. Dataset will only return images.")
            self.mask_files = [None] * len(self.image_files) # Create dummy list
        elif len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) do not match.")

        # Basic check for filename correspondence (optional, but good practice)
        # for img_f, msk_f in zip(self.image_files, self.mask_files):
        #     if msk_f and os.path.basename(img_f) != os.path.basename(msk_f):
        #         print(f"Warning: Image {os.path.basename(img_f)} and mask {os.path.basename(msk_f)} might not correspond.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        try:
            # image = Image.open(img_path).convert("RGB") # Using PIL
            # For preprocessor, it expects path or BGR numpy array
            image_tensor = self.preprocessor.preprocess(img_path)

            if mask_path:
                # mask = Image.open(mask_path) # .convert("L") for grayscale labels
                mask_tensor = self.preprocessor.preprocess_mask(mask_path, self.num_classes)
            else:
                # Return a dummy mask if no mask path (e.g., for inference)
                # The shape should match preprocessor's output mask shape
                dummy_mask_shape = self.preprocessor.resize_dim # (H, W)
                mask_tensor = torch.zeros(dummy_mask_shape, dtype=torch.long)

            return image_tensor, mask_tensor
        
        except Exception as e:
            print(f"Error loading item {idx} (image: {img_path}, mask: {mask_path}): {e}")
            # Return None or raise error, or return a dummy item
            # For robustness, one might try loading the next item or returning a dummy
            # For simplicity here, we'll let it crash or return problematic data that
            # a collate_fn might need to handle.
            # A robust solution might involve a custom collate_fn that filters Nones.
            dummy_img_tensor = torch.zeros((3, self.preprocessor.resize_dim[0], self.preprocessor.resize_dim[1]))
            dummy_mask_tensor = torch.zeros(self.preprocessor.resize_dim, dtype=torch.long)
            return dummy_img_tensor, dummy_mask_tensor

# Example of how to use it (would be in train script)
if __name__ == '__main__': # In a separate file or guarded in train script
    # This __main__ block is for testing the dataset class itself.
    # You'd need to create dummy data or point to your actual dataset.
    
    # Create dummy data structure for testing
    base_dir = "temp_mango_dataset"
    train_img_dir = os.path.join(base_dir, "train", "images")
    train_mask_dir = os.path.join(base_dir, "train", "masks")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)

    # Create a few dummy images and masks
    for i in range(3):
        dummy_img_arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(train_img_dir, f"image_{i}.png"), dummy_img_arr)
        
        dummy_mask_arr = np.random.randint(0, 2, (200, 200), dtype=np.uint8) # Binary mask
        cv2.imwrite(os.path.join(train_mask_dir, f"image_{i}.png"), dummy_mask_arr) # Same name

    # Assuming ImagePreprocessor is defined in 'MinGraph_UNet.preprocessing.image_preprocessing.image_preprocess'
    # For this test, let's use a mock preprocessor or define it here if not importable directly
    class MockImagePreprocessor:
        def __init__(self, resize_dim=(128,128)):
            self.resize_dim = resize_dim
            self.transform_img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(resize_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        def preprocess(self, img_path_or_arr):
            img = cv2.imread(img_path_or_arr)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform_img(img_rgb)
        def preprocess_mask(self, mask_path_or_arr, num_classes):
            mask = cv2.imread(mask_path_or_arr, cv2.IMREAD_GRAYSCALE)
            mask_r = cv2.resize(mask, (self.resize_dim[1], self.resize_dim[0]), interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy(mask_r).long()

    mock_prep = MockImagePreprocessor()
    
    try:
        dataset = MangoDataset(image_dir=train_img_dir,
                               mask_dir=train_mask_dir,
                               preprocessor=mock_prep,
                               num_classes=2) # BG, Mango
        print(f"Dataset size: {len(dataset)}")
        img_tensor, mask_tensor = dataset[0]
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Mask tensor shape: {mask_tensor.shape}")
        print(f"Mask tensor dtype: {mask_tensor.dtype}")
    except Exception as e:
        print(f"Error during dataset test: {e}")
    finally:
        # Clean up dummy directory
        import shutil
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            print(f"Removed temporary dataset directory: {base_dir}")