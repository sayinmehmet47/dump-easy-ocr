import easyocr
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

class OCRReader:
    def __init__(
        self,
        languages: List[str] = ['de', 'fr', 'it'],
        gpu: bool = True,
        model_storage_directory: Optional[str] = None,
        download_enabled: bool = True,
        text_threshold: float = 0.7,
        paragraph: bool = False
    ):
        """
        Initialize the OCR reader with configurable parameters.
        
        Args:
            languages: List of language codes to detect
            gpu: Whether to use GPU
            model_storage_directory: Directory to store models
            download_enabled: Whether to allow downloading models
            text_threshold: Threshold for text recognition
            paragraph: Whether to group text into paragraphs
        """
        self.reader = easyocr.Reader(
            lang_list=languages,
            gpu=gpu,
            model_storage_directory=model_storage_directory,
            download_enabled=download_enabled,
            recog_network='standard'
        )
        self.text_threshold = text_threshold
        self.paragraph = paragraph

    def resize_image(self, image: np.ndarray, max_size: int = 1000) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        """
        height, width = image.shape[:2]
        
        if height <= max_size and width <= max_size:
            return image
        
        ratio = max_size / float(max(height, width))
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def process_image_ocr(
        self,
        image: np.ndarray,
        min_confidence: float = 0.0,
        resize_max: int = 1000
    ) -> Dict[str, Any]:
        """
        Process an image through OCR and return the results.
        
        Args:
            image: numpy array of the image
            min_confidence: Minimum confidence threshold for results
            resize_max: Maximum dimension for image resizing
        """
        original_height, original_width = image.shape[:2]
        resized_image = self.resize_image(image, max_size=resize_max)
        resize_ratio = original_width / resized_image.shape[1]
        
        # Use paragraph mode if configured
        results = self.reader.readtext(
            resized_image,
            paragraph=self.paragraph,
            text_threshold=self.text_threshold
        )
        
        processed_results = []
        for result in results:
            confident = float(result[2])
            
            # Skip results below confidence threshold
            if confident < min_confidence:
                continue
                
            boxes = []
            for point in result[0]:
                x = int(point[0] * resize_ratio)
                y = int(point[1] * resize_ratio)
                boxes.append([x, y])
                
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
