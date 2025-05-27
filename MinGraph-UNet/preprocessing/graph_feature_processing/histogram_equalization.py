import cv2
import numpy as np

class HistogramEqualizer:
    def __init__(self):
        """Initializes the HistogramEqualizer."""
        pass

    def equalize_histogram_rgb(self, image_array_rgb):
        """
        Applies histogram equalization to an RGB image.
        Commonly done by converting to a colorspace like YUV or HSV,
        equalizing the intensity/luminance channel, and converting back.

        Args:
            image_array_rgb (np.ndarray): Input image as a NumPy array (H, W, C) in RGB format.

        Returns:
            np.ndarray: Histogram equalized RGB image (H, W, C).
        """
        if image_array_rgb.ndim != 3 or image_array_rgb.shape[2] != 3:
            raise ValueError("Input image must be an RGB image (H, W, 3).")

        # Convert RGB to YUV (Y is luminance, U and V are chrominance)
        img_yuv = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2YUV)

        # Equalize the Y channel (luminance)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # Convert YUV back to RGB
        equalized_rgb_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        return equalized_rgb_image
    
    def equalize_histogram_gray(self, image_array_gray):
        """
        Applies histogram equalization to a grayscale image.

        Args:
            image_array_gray (np.ndarray): Input image as a NumPy array (H, W) in grayscale.

        Returns:
            np.ndarray: Histogram equalized grayscale image (H, W).
        """
        if image_array_gray.ndim != 2:
            raise ValueError("Input image must be a grayscale image (H, W).")
        
        equalized_gray_image = cv2.equalizeHist(image_array_gray)
        return equalized_gray_image


if __name__ == '__main__':
    # Create a dummy low-contrast RGB image
    dummy_rgb_image = np.full((100, 100, 3), 100, dtype=np.uint8) # Dark gray
    dummy_rgb_image[25:75, 25:75, :] = 150 # Slightly lighter gray square
    cv2.imwrite("dummy_low_contrast_rgb.png", cv2.cvtColor(dummy_rgb_image, cv2.COLOR_RGB2BGR))


    equalizer = HistogramEqualizer()
    equalized_image = equalizer.equalize_histogram_rgb(dummy_rgb_image)

    print("Equalized image shape:", equalized_image.shape)
    print("Equalized image dtype:", equalized_image.dtype)
    
    # Display or save the result (optional)
    cv2.imwrite("dummy_equalized_rgb.png", cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR))
    print("Saved dummy_low_contrast_rgb.png and dummy_equalized_rgb.png")

    # Test grayscale
    dummy_gray_image = cv2.cvtColor(dummy_rgb_image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("dummy_low_contrast_gray.png", dummy_gray_image)
    equalized_gray = equalizer.equalize_histogram_gray(dummy_gray_image)
    cv2.imwrite("dummy_equalized_gray.png", equalized_gray)
    print("Saved dummy_low_contrast_gray.png and dummy_equalized_gray.png")


    # Clean up
    os.remove("dummy_low_contrast_rgb.png")
    os.remove("dummy_equalized_rgb.png")
    os.remove("dummy_low_contrast_gray.png")
    os.remove("dummy_equalized_gray.png")