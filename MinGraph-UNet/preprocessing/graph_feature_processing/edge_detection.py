import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, kernel_size=3):
        """
        Initializes the EdgeDetector.

        Args:
            kernel_size (int): Kernel size for the Sobel operator (e.g., 3, 5).
        """
        self.kernel_size = kernel_size

    def sobel_edges(self, image_array_rgb):
        """
        Applies the Sobel operator to detect edges in an image.

        Args:
            image_array_rgb (np.ndarray): Input image as a NumPy array (H, W, C) in RGB format.

        Returns:
            np.ndarray: Edge magnitude map (H, W), normalized to [0, 255] uint8.
        """
        if image_array_rgb.ndim != 3 or image_array_rgb.shape[2] != 3:
            raise ValueError("Input image must be an RGB image (H, W, 3).")

        # Convert to grayscale for edge detection
        gray_image = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2GRAY)

        # Apply Sobel operator in X and Y directions
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.kernel_size)

        # Compute the magnitude of gradients
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize to 0-255 and convert to uint8
        # Note: cv2.convertScaleAbs also does this effectively
        if np.max(edge_magnitude) > 0:
             edge_magnitude_normalized = (edge_magnitude / np.max(edge_magnitude) * 255).astype(np.uint8)
        else: # Handle case of all-zero magnitude (e.g. solid color image)
            edge_magnitude_normalized = np.zeros_like(edge_magnitude, dtype=np.uint8)

        return edge_magnitude_normalized

if __name__ == '__main__':
    # Create a dummy RGB image
    dummy_rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a white square in the middle to create edges
    dummy_rgb_image[25:75, 25:75, :] = 255 
    
    cv2.imwrite("dummy_rgb_for_edge.png", cv2.cvtColor(dummy_rgb_image, cv2.COLOR_RGB2BGR))


    detector = EdgeDetector(kernel_size=3)
    edge_map = detector.sobel_edges(dummy_rgb_image)

    print("Edge map shape:", edge_map.shape) # Expected: (100, 100)
    print("Edge map dtype:", edge_map.dtype) # Expected: uint8
    print("Edge map min/max:", np.min(edge_map), np.max(edge_map))

    # Display or save the result (optional)
    cv2.imwrite("dummy_edge_map.png", edge_map)
    print("Saved dummy_rgb_for_edge.png and dummy_edge_map.png")

    # Clean up
    os.remove("dummy_rgb_for_edge.png")
    os.remove("dummy_edge_map.png")