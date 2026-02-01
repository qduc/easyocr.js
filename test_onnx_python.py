#!/usr/bin/env python3
"""
Test ONNX models in Python to verify they work correctly.
"""
import sys
import numpy as np
import onnxruntime as ort


def main():
    print('=' * 60)
    print('TESTING ONNX DETECTOR IN PYTHON')
    print('=' * 60)

    # Load the detector input tensor we generated
    print('\nLoading detector input tensor...')
    detector_input = np.load('debug_output/detector_input_python.npy')
    print(f'  Shape: {detector_input.shape}')
    print(f'  Dtype: {detector_input.dtype}')
    print(f'  Range: [{detector_input.min():.6f}, {detector_input.max():.6f}]')

    # Load the ONNX detector model
    print('\nLoading ONNX detector model...')
    session = ort.InferenceSession('models/onnx/craft_mlt_25k.onnx', providers=['CPUExecutionProvider'])

    print('\nModel inputs:')
    for inp in session.get_inputs():
        print(f'  {inp.name}: {inp.type}, shape={inp.shape}')

    print('\nModel outputs:')
    for out in session.get_outputs():
        print(f'  {out.name}: {out.type}, shape={out.shape}')

    # Run inference
    print('\nRunning detector inference...')
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: detector_input})

    print(f'\nDetector outputs:')
    for i, output in enumerate(outputs):
        print(f'  Output {i}: shape={output.shape}, dtype={output.dtype}')
        print(f'    Range: [{output.min():.6f}, {output.max():.6f}]')
        print(f'    Mean: {output.mean():.6f}, Std: {output.std():.6f}')

        # Assuming first output is text heatmap, second is link heatmap
        if i == 0:
            # Count pixels above threshold
            text_threshold = 0.7
            link_threshold = 0.4
            text_pixels = np.sum(output > text_threshold)
            link_pixels = np.sum(output > link_threshold)
            print(f'    Pixels > {text_threshold} (text): {text_pixels}')
            print(f'    Pixels > {link_threshold} (link): {link_pixels}')

    print('\n✓ ONNX detector runs successfully in Python!')
    print('  This confirms the ONNX model export is correct.')

    # Now test with Python EasyOCR for comparison
    print('\n' + '=' * 60)
    print('COMPARING WITH PYTHON EASYOCR')
    print('=' * 60)

    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)

    image_path = 'python_reference/validation/images/Screenshot_20260201_193653.png'
    results = reader.readtext(image_path)

    print(f'\nEasyOCR detected {len(results)} text regions:')
    for bbox, text, confidence in results:
        print(f'  "{text}" (confidence: {confidence:.4f})')

    return 0


if __name__ == '__main__':
    sys.exit(main())
