import easyocr
import numpy as np

def process_image_ocr(image):
    """
    Process an image through OCR and return the results.
    Args:
        image: numpy array of the image
    Returns:
        dict: OCR results in the specified format with image size and detected text
    """
    reader = easyocr.Reader(['de', 'fr', 'it'], gpu=True, download_enabled=True)
    results = reader.readtext(image)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Process OCR results
    processed_results = []
    for result in results:
        # Convert coordinates from numpy arrays to lists
        boxes = [[int(x), int(y)] for x, y in result[0]]
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
            'width': width,
            'height': height
        }
    }
