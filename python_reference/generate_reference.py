import easyocr
import json
import os
import sys

def generate_reference(image_path, output_json):
    reader = easyocr.Reader(['en']) # build the model
    results = reader.readtext(image_path)

    # Format results: list of [box, text, confidence]
    # box is list of [x, y] coordinates
    formatted_results = []
    for (bbox, text, prob) in results:
        formatted_results.append({
            "box": [[float(coord[0]), float(coord[1])] for coord in bbox],
            "text": text,
            "confidence": float(prob)
        })

    with open(output_json, 'w') as f:
        json.dump(formatted_results, f, indent=2)
    print(f"Reference saved to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_reference.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = f"{name}_reference.json"
    generate_reference(img_path, output_path)
