import json
from functools import lru_cache

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title='easyocr.js Python reference')


@lru_cache(maxsize=8)
def _get_reader(langs_key: str, gpu: bool) -> easyocr.Reader:
    # Cache readers; init is expensive.
    langs = json.loads(langs_key)
    return easyocr.Reader(langs, gpu=gpu)


def _format_results(results):
    return [
        {
            'box': [[float(pt[0]), float(pt[1])] for pt in bbox],
            'text': text,
            'confidence': float(prob),
        }
        for (bbox, text, prob) in results
    ]


@app.get('/health')
def health():
    return {'ok': True}


@app.post('/readtext')
async def readtext(
    image: UploadFile = File(...),
    lang: list[str] = Query(default=['en']),
    gpu: bool = Query(default=False),
):
    content = await image.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    mat = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if mat is None:
        return JSONResponse(status_code=400, content={'error': 'Unable to decode image'})

    reader = _get_reader(json.dumps(lang), gpu)
    results = reader.readtext(mat)

    return {
        'formatVersion': 1,
        'image': image.filename,
        'results': _format_results(results),
    }

