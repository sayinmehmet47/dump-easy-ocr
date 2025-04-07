from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import cv2
import numpy as np
from ocr_reader import OCRReader
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRConfig(BaseModel):
    languages: List[str] = Field(default=['de', 'fr', 'it'], description="List of languages to detect")
    gpu: bool = Field(default=True, description="Whether to use GPU for processing")
    model_storage_directory: Optional[str] = Field(default=None, description="Directory to store models")
    download_enabled: bool = Field(default=True, description="Whether to allow downloading models")
    detector_threshold: float = Field(default=0.5, description="Threshold for text detection", ge=0, le=1)
    text_threshold: float = Field(default=0.7, description="Threshold for text recognition", ge=0, le=1)
    paragraph: bool = Field(default=False, description="Whether to group text into paragraphs")
    min_confidence: float = Field(default=0.0, description="Minimum confidence threshold for results", ge=0, le=1)
    resize_max: int = Field(default=1000, description="Maximum dimension for image resizing", gt=0)

app = FastAPI(
    title="Health Insurance Card OCR API",
    description="API for extracting information from German, French, and Italian health insurance cards",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store OCR reader instance
ocr_reader = None

@app.get("/")
async def root():
    return {
        "message": "Health Insurance Card OCR API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.post("/configure")
async def configure_ocr(config: OCRConfig):
    """Configure the OCR reader with new parameters"""
    global ocr_reader
    try:
        ocr_reader = OCRReader(
            languages=config.languages,
            gpu=config.gpu,
            model_storage_directory=config.model_storage_directory,
            download_enabled=config.download_enabled,
            detector_threshold=config.detector_threshold,
            text_threshold=config.text_threshold,
            paragraph=config.paragraph
        )
        return {"message": "OCR configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error configuring OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error configuring OCR: {str(e)}")

@app.post("/readtext")
async def read_text(request: Request, config: Optional[OCRConfig] = None):
    """
    Process image with OCR using provided configuration.
    If no configuration is provided, uses default or last configured settings.
    """
    try:
        global ocr_reader
        
        # Initialize OCR reader if not exists or new config provided
        if ocr_reader is None or config is not None:
            config = config or OCRConfig()
            ocr_reader = OCRReader(
                languages=config.languages,
                gpu=config.gpu,
                model_storage_directory=config.model_storage_directory,
                download_enabled=config.download_enabled,
                detector_threshold=config.detector_threshold,
                text_threshold=config.text_threshold,
                paragraph=config.paragraph
            )
        
        # Read the raw image data
        contents = await request.body()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Convert to OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process image with OCR
        min_confidence = config.min_confidence if config else 0.0
        resize_max = config.resize_max if config else 1000
        
        results = ocr_reader.process_image_ocr(
            image,
            min_confidence=min_confidence,
            resize_max=resize_max
        )
        
        if not results['results']:
            return JSONResponse(
                status_code=422,
                content={"error": "No text detected in the image"}
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the image: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3755, log_level="debug")