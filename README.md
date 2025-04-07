# Health Insurance Card OCR API

A FastAPI-based OCR service that extracts text from health insurance cards in German, French, and Italian using EasyOCR.

## Features

- Multi-language support (German, French, Italian)
- Configurable OCR parameters
- GPU acceleration support
- REST API with Swagger documentation
- Adjustable confidence thresholds
- Image preprocessing capabilities

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
pip install fastapi uvicorn easyocr opencv-python numpy pydantic python-multipart
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

### 1. Configure OCR Settings

You can configure the OCR settings using the `/configure` endpoint:

```bash
curl -X POST "http://localhost:3755/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "languages": ["de", "fr", "it"],
    "gpu": true,
    "detector_threshold": 0.5,
    "text_threshold": 0.7,
    "min_confidence": 0.3
  }'
```

### 2. Process Images

#### Option 1: Using default configuration

```bash
curl -X POST "http://localhost:3755/readtext" \
  --data-binary @path/to/your/image.jpg
```

#### Option 2: With custom configuration for this request

```bash
curl -X POST "http://localhost:3755/readtext" \
  -F "image=@path/to/your/image.jpg" \
  -F 'config={
    "languages": ["de"],
    "min_confidence": 0.6,
    "gpu": false
  }'
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `languages` | List[str] | `["de", "fr", "it"]` | Languages to detect |
| `gpu` | bool | `true` | Whether to use GPU acceleration |
| `model_storage_directory` | str | `null` | Custom directory to store models |
| `download_enabled` | bool | `true` | Allow downloading models |
| `detector_threshold` | float | `0.5` | Text detection confidence threshold (0-1) |
| `text_threshold` | float | `0.7` | Text recognition confidence threshold (0-1) |
| `paragraph` | bool | `false` | Group text into paragraphs |
| `min_confidence` | float | `0.0` | Minimum confidence for returned results (0-1) |
| `resize_max` | int | `1000` | Maximum image dimension for processing |

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

1. Use GPU acceleration when possible
2. Adjust `resize_max` for better performance on large images
3. Set appropriate confidence thresholds to filter out low-quality results
4. Use specific languages instead of all languages when possible

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
