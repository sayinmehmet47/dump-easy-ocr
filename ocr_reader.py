import easyocr
import numpy as np
import cv2

def resize_image(image, max_size=1000):
    """
    Resize image while maintaining aspect ratio so that the largest dimension
    does not exceed max_size.
    """
    height, width = image.shape[:2]
    
    # If image is already smaller than max_size, return original
    if height <= max_size and width <= max_size:
        return image
    
    # Calculate the ratio
    ratio = max_size / float(max(height, width))
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def process_image_ocr(image):
    """
    Process an image through OCR and return the results.
    Args:
        image: numpy array of the image
    Returns:
        dict: OCR results in the specified format with image size and detected text
    """
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Resize image for faster processing
    resized_image = resize_image(image)
    resize_ratio = original_width / resized_image.shape[1]  # Calculate resize ratio
    
    reader = easyocr.Reader(['de', 'fr', 'it'], gpu=True, download_enabled=True)
    results = reader.readtext(resized_image)
    
    # Process OCR results
    processed_results = []
    for result in results:
        # Scale the coordinates back to original image size
        boxes = [[int(x * resize_ratio), int(y * resize_ratio)] for x, y in result[0]]
        # Convert confidence to float
        confident = float(result[2])
        processed_results.append({
            'text': result[1],
            'boxes': boxes,
            'confident': confident
        })
    
    return {
        'results': processed_results,
        'imageSize': {
            'width': original_width,
            'height': original_height
        }
    }
