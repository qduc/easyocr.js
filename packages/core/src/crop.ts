import type { Box, OcrOptions, RasterImage } from './types.js';
import { cropBox, rotateBox, rotateImage, warpPerspective } from './utils.js';

export interface CropResult {
  image: RasterImage;
  box: Box;
  rotation: number;
}

export const cropHorizontal = (image: RasterImage, boxes: Box[]): CropResult[] =>
  boxes.map((box) => ({ image: cropBox(image, box), box, rotation: 0 }));

export const cropFree = (image: RasterImage, boxes: Box[]): CropResult[] => {
  return boxes.map((box) => {
    const xs = box.map((point) => point[0]);
    const ys = box.map((point) => point[1]);
    const width = Math.max(1, Math.round(Math.max(...xs) - Math.min(...xs)));
    const height = Math.max(1, Math.round(Math.max(...ys) - Math.min(...ys)));
    return { image: warpPerspective(image, box, width, height).image, box, rotation: 0 };
  });
};

export const applyRotationSearch = (
  image: RasterImage,
  box: Box,
  rotations: number[],
): { image: RasterImage; box: Box; rotation: number }[] => {
  if (!rotations.length) {
    return [{ image, box, rotation: 0 }];
  }
  return rotations.map((rotation) => ({
    image: rotateImage(image, rotation),
    box: rotateBox(box, rotation, image.width, image.height),
    rotation,
  }));
};

export const buildCrops = (
  image: RasterImage,
  horizontalList: Box[],
  freeList: Box[],
  options: OcrOptions,
): CropResult[] => {
  const horizontal = cropHorizontal(image, horizontalList);
  const free = cropFree(image, freeList);
  if (!options.rotationInfo.length) {
    return [...horizontal, ...free];
  }
  const rotated: CropResult[] = [];
  for (const crop of [...horizontal, ...free]) {
    for (const variant of applyRotationSearch(crop.image, crop.box, options.rotationInfo)) {
      rotated.push({ ...variant, rotation: variant.rotation });
    }
  }
  return rotated;
};
