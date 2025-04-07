#!/bin/bash
cd /Users/flow/dump-easy-ocr
export PYTHONPATH="/Users/flow/Library/Python/3.9/lib/python/site-packages:$PYTHONPATH"
/usr/bin/python3 -m uvicorn api:app --host 0.0.0.0 --port 3755 --workers 4