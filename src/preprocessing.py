"""
Preprocessing utilities for CSE480 Machine Vision Project
Handles image preprocessing for emotion recognition model.
"""

import cv2
import numpy as np


def process_face(image):
    """
    Processes a face image for emotion recognition model.
    
    Converts the image to grayscale and resizes it to 48x48 pixels
    as required by the Emotion Model specifications.
    
    Args:
        image: Input image (numpy array) in BGR, RGB, or grayscale format.
               Can be a file path (str) or numpy array.
    
    Returns:
        numpy.ndarray: Processed grayscale image of shape (48, 48)
    """
    # Load image if it's a file path
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from path: {image}")
    else:
        img = image.copy()
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        # Image is BGR or RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        gray = img
    
    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    return resized


if __name__ == "__main__":
    # Create a dummy blank image (e.g., 100x100 grayscale)
    dummy_image = np.zeros((100, 100), dtype=np.uint8)
    
    # Process the image
    processed = process_face(dummy_image)
    
    # Print the final shape
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {processed.shape}")
    
    # Verify it's (48, 48)
    assert processed.shape == (48, 48), f"Expected shape (48, 48), got {processed.shape}"
    print("\n✓ Test passed! Output shape is (48, 48)")

