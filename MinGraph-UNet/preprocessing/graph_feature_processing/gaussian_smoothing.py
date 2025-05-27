import cv2
import numpy as np

class GaussianSmoother:
    def __init__(self, kernel_size=(5, 5), sigma_x=1.0, sigma_y=None):
        """
        Initializes the GaussianSmoother.

        Args:
            kernel_size (tuple): Gaussian kernel size (width, height). Must be odd and positive.
            sigma_x (float): Gaussian kernel standard deviation in X direction.
            sigma_y (float, optional): Gaussian kernel standard deviation in Y direction.
                                       If None, it is set to be the same as sigma_x.
        """
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y if sigma_y is not None else sigma_x

        if not (self.kernel_size[0] > 0 and self.kernel_size[0] % 2 == 1 and \
                self.kernel_size[1] > 0 and self.kernel_size[1] % 2 == 1):
            raise ValueError("Kernel size must be positive and odd.")

    def smooth(self, image_array):
        """
        Applies Gaussian smoothing to an image.

        Args:
            image_array (np.ndarray): Input image as a NumPy array (H, W) or (H, W, C).

        Returns:
            np.ndarray: Smoothed image.
        """
        smoothed_image = cv2.GaussianBlur(image_array, self.kernel_size, self.sigma_x, sigmaY=self.sigma_y)
        return smoothed_image

if __name__ == '__main__':
    # Create a dummy image with some noise
    dummy_image = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
    dummy_image[25:75, 25:75, :] += 100 # Add a brighter region
    dummy_image = np.clip(dummy_image, 0, 255)
    cv2.imwrite("dummy_noisy_image.png", cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR if dummy_image.ndim ==3 else None))


    smoother = GaussianSmoother(kernel_size=(5, 5), sigma_x=1.5)
    smoothed_image = smoother.smooth(dummy_image)

    print("Smoothed image shape:", smoothed_image.shape)
    print("Smoothed image dtype:", smoothed_image.dtype)
    
    cv2.imwrite("dummy_smoothed_image.png", cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2BGR if smoothed_image.ndim ==3 else None))
    print("Saved dummy_noisy_image.png and dummy_smoothed_image.png")

    # Test grayscale
    dummy_gray_image = cv2.cvtColor(dummy_image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("dummy_noisy_gray.png", dummy_gray_image)
    smoothed_gray = smoother.smooth(dummy_gray_image)
    cv2.imwrite("dummy_smoothed_gray.png", smoothed_gray)
    print("Saved dummy_noisy_gray.png and dummy_smoothed_gray.png")

    # Clean up
    os.remove("dummy_noisy_image.png")
    os.remove("dummy_smoothed_image.png")
    os.remove("dummy_noisy_gray.png")
    os.remove("dummy_smoothed_gray.png")