# Health Insurance Card OCR API

A FastAPI-based OCR service that extracts text from health insurance cards in German, French, and Italian using EasyOCR.

## Features

- Multi-language support (German, French, Italian)
- Configurable OCR parameters via query parameters
- GPU acceleration support
- REST API with Swagger documentation
- Adjustable confidence thresholds
- Image preprocessing capabilities
- Efficient caching of OCR instances for better performance

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for better performance)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd easy-ocr
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install fastapi uvicorn easyocr opencv-python numpy
```

## Running the API

Start the API server:

```bash
python api.py
```

The server will start at `http://localhost:3755` with the following endpoints:

- API documentation: `http://localhost:3755/docs`
- Alternative documentation: `http://localhost:3755/redoc`

## API Usage

### Process Images

The API provides a single endpoint `/readtext` that accepts both the image and configuration parameters. You can use it in different ways:

#### 1. Basic Usage (Default Configuration)

```bash
curl -X POST "http://localhost:3755/readtext" \
  -H "Content-Type: image/png" \
  --data-binary "@path/to/your/image.png"
```

#### 2. Custom Language Configuration

```bash
curl -X POST "http://localhost:3755/readtext?languages=de,fr" \
  -H "Content-Type: image/png" \
  --data-binary "@path/to/your/image.png"
```

#### 3. Advanced Configuration

```bash
curl -X POST "http://localhost:3755/readtext?languages=de,fr&text_threshold=0.8&min_confidence=0.6&resize_max=800" \
  -H "Content-Type: image/png" \
  --data-binary "@path/to/your/image.png"
```

#### 4. Disable GPU (CPU-only mode)

```bash
curl -X POST "http://localhost:3755/readtext?gpu=false" \
  -H "Content-Type: image/png" \
  --data-binary "@path/to/your/image.png"
```

## Configuration Options

All configuration options are passed as query parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `languages` | string | `"de,fr,it"` | Comma-separated list of languages to detect |
| `gpu` | boolean | `true` | Whether to use GPU acceleration |
| `model_storage_directory` | string | `null` | Custom directory to store models |
| `download_enabled` | boolean | `true` | Allow downloading models |
| `text_threshold` | float | `0.7` | Text recognition confidence threshold (0-1) |
| `paragraph` | boolean | `false` | Group text into paragraphs |
| `min_confidence` | float | `0.0` | Minimum confidence for returned results (0-1) |
| `resize_max` | integer | `1000` | Maximum image dimension for processing |

## API Response Format

The API returns JSON responses in the following format:

```json
{
  "results": [
    {
      "text": "Detected text",
      "boxes": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "confident": 0.95
    }
  ],
  "imageSize": {
    "width": 800,
    "height": 600
  }
}
```

- `text`: The detected text
- `boxes`: Coordinates of the bounding box (4 corners)
- `confident`: Confidence score (0-1)
- `imageSize`: Original image dimensions

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Successful operation
- `400`: Invalid request (e.g., empty or invalid image)
- `422`: No text detected in image
- `500`: Server error

## Performance Tips

1. The API implements caching of OCR instances:
   - First request with a specific configuration will take longer (~4 seconds)
   - Subsequent requests with the same configuration will be faster (~1 second)
   - Changing configuration parameters will create a new cached instance
2. Use GPU acceleration when possible
3. Adjust `resize_max` for better performance on large images
4. Set appropriate confidence thresholds to filter out low-quality results
5. Use specific languages instead of all languages when possible

## Development

The project structure:

```
easy-ocr/
├── api.py           # FastAPI application and endpoints
├── ocr_reader.py    # OCR processing logic
└── README.md        # Documentation
```

## License

[Your chosen license]
