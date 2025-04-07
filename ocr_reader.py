def process_image_ocr(image):
    """
    Process an image through OCR and return the results.
    Args:
        image: numpy array of the image
    Returns:
        results: list of OCR results
    """
    reader = easyocr.Reader(['de', 'fr', 'it'], gpu=True, download_enabled=True)
    results = reader.readtext(image)
    return results
