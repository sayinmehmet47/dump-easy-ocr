# Health Insurance Card OCR API

API service for extracting text from German, French, and Italian health insurance cards using EasyOCR.

## Features

- Support for German, French, and Italian text recognition
- Fast processing using GPU acceleration (when available)
- Simple REST API interface

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running with Docker

```bash
docker build -t health-card-ocr .
docker run -p 3755:3755 health-card-ocr
```

## API Endpoints

### GET /

Returns API information and supported languages.

### POST /readtext

Processes an image and returns detected text.

## Usage Examples

### Using cURL

Send an image for processing:

```bash
curl -X POST \
  -H "Content-Type: image/png" \
  --data-binary "@example.png" \
  http://api-address/readtext
```

### Example Response

```json
{
  "status": "success",
  "results": [
    {
      "text": "Detected text here",
      "confidence": 0.95
    }
  ]
}
```

## Error Responses

- `400`: Invalid or empty image
- `422`: No text detected in image
- `500`: Server processing error

## Supported Image Formats

- PNG
- JPEG
- JPG
