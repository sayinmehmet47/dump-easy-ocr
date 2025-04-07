from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ocr_reader import process_image_ocr
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
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {
        "message": "Health Insurance Card OCR API",
    }

@app.post("/readtext")
async def read_text(request: Request):
    try:
        logger.debug("Received request for raw image processing")
        
        # Read the raw image data from request body
        contents = await request.body()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Convert to OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process image with OCR
        results = process_image_ocr(image)
        
        if not results:
            return JSONResponse(
                status_code=422,
                content={"error": "No text detected in the image"}
            )
        
        # Return results directly without the status wrapper
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