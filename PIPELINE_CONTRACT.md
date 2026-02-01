# Pipeline Contract

This document defines the interface and transformations at each stage of the EasyOCR pipeline.

## 1. Preprocessing
- Input: Raw image (Buffer, File, or Canvas)
- Normalization: RGB scaling, resizing (maintaining aspect ratio or padding).
- Output: Tensor [1, 3, H, W]

## 2. Detector
- Model: CRAFT (Scene Text Detection)
- Input: Preprocessed image tensor
- Output: Heatmaps (Region score, Affinity score)

## 3. Box Post-processing
- Transformation: Thresholding heatmaps -> Connected components -> Bounding boxes (Quadrilaterals).
- Output: List of polygons/boxes.

## 4. Cropping & Deskewing
- For each box:
  - Crop from original image.
  - Perspective transform to rectify/flatten the text area.
  - Greyscale conversion (optional? EasyOCR usually recognizes on grey or RGB depending on model).
- Output: List of cropped text line images.

## 5. Recognizer
- Model: ResNet + BiLSTM + CTC (standard)
- Input: Resized text line crops (e.g., 100px width, 32px height)
- Output: Logits/Character probabilities.

## 6. Decoding
- Method: Greedy CTC decoding or Beam Search.
- Output: String per crop.

## 7. Result Merging
- Combine boxes, text strings, and confidence scores.
- Return final JSON object.
