from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import cv2
import numpy as np
from ocr_reader import OCRReader, OCRConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Health Insurance Card OCR API",
    description="API for extracting information from German, French, and Italian health insurance cards",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for OCRReader instances
reader_cache: Dict[str, OCRReader] = {}

def get_cache_key(config_params: dict) -> str:
    """Generate a cache key from configuration parameters"""
    # Only include parameters that affect reader initialization
    key_params = {
        'languages': config_params['languages'],
        'gpu': config_params['gpu'],
        'model_storage_directory': config_params['model_storage_directory'],
        'download_enabled': config_params['download_enabled'],
        'text_threshold': config_params['text_threshold'],
        'paragraph': config_params['paragraph']
    }
    return str(sorted(key_params.items()))

def get_or_create_reader(config_params: dict) -> OCRReader:
    """Get an existing OCRReader instance or create a new one if needed"""
    cache_key = get_cache_key(config_params)
    
    if cache_key not in reader_cache:
        logger.debug("Creating new OCRReader instance")
        reader_cache[cache_key] = OCRReader(
            languages=[lang.strip() for lang in config_params['languages'].split(",")],
            gpu=config_params['gpu'],
            model_storage_directory=config_params['model_storage_directory'],
            download_enabled=config_params['download_enabled'],
            text_threshold=config_params['text_threshold'],
            paragraph=config_params['paragraph']
        )
    else:
        logger.debug("Using cached OCRReader instance")
    
    return reader_cache[cache_key]

@app.get("/")
async def root():
    return {
        "message": "Health Insurance Card OCR API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.post("/readtext")
async def read_text(
    request: Request,
    languages: str = Query(default=",".join(OCRConfig.DEFAULT_LANGUAGES), description="Comma-separated list of languages to detect"),
    gpu: bool = Query(default=OCRConfig.DEFAULT_GPU, description="Whether to use GPU for processing"),
    model_storage_directory: Optional[str] = Query(default=None, description="Directory to store models"),
    download_enabled: bool = Query(default=OCRConfig.DEFAULT_DOWNLOAD_ENABLED, description="Whether to allow downloading models"),
    text_threshold: float = Query(default=OCRConfig.DEFAULT_TEXT_THRESHOLD, ge=0, le=1, description="Threshold for text recognition"),
    paragraph: bool = Query(default=OCRConfig.DEFAULT_PARAGRAPH, description="Whether to group text into paragraphs"),
    min_confidence: float = Query(default=OCRConfig.DEFAULT_MIN_CONFIDENCE, ge=0, le=1, description="Minimum confidence threshold for results"),
    resize_max: int = Query(default=OCRConfig.DEFAULT_RESIZE_MAX, gt=0, description="Maximum dimension for image resizing")
):
    """
    Process image with OCR using provided configuration.
    Accepts raw image data in the request body and configuration via query parameters.
    Example: POST /readtext?languages=de,fr&text_threshold=0.8
    """
    try:
        # Get configuration parameters
        config_params = {
            'languages': languages,
            'gpu': gpu,
            'model_storage_directory': model_storage_directory,
            'download_enabled': download_enabled,
            'text_threshold': text_threshold,
            'paragraph': paragraph,
            'min_confidence': min_confidence,
            'resize_max': resize_max
        }
        
        # Get or create OCR reader instance
        ocr_reader = get_or_create_reader(config_params)
        
        # Read the raw image data
        contents = await request.body()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Convert to OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process image with OCR using configuration values
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