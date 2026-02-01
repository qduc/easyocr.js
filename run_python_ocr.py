import easyocr
import sys
import numpy as np
from pathlib import Path

image_path = sys.argv[1]
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models', download_enabled=False)
results = reader.readtext(image_path, paragraph=True)

import json
# Convert results to a serializable format
serializable = []
for item in results:
    if len(item) == 3:
        box, text, conf = item
    else:
        box, text = item
        conf = 1.0
    serializable.append({
        "box": [list(p) for p in box] if isinstance(box, (list, tuple, np.ndarray)) else box,
        "text": text,
        "confidence": float(conf)
    })

print(json.dumps(serializable, indent=2))
