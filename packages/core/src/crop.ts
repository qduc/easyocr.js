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
    const tl = box[0];
    const tr = box[1];
    const br = box[2];
    const bl = box[3];
    const widthA = Math.hypot(br[0] - bl[0], br[1] - bl[1]);
    const widthB = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
    const heightA = Math.hypot(tr[0] - br[0], tr[1] - br[1]);
    const heightB = Math.hypot(tl[0] - bl[0], tl[1] - bl[1]);
    const width = Math.max(1, Math.trunc(Math.max(widthA, widthB)));
    const height = Math.max(1, Math.trunc(Math.max(heightA, heightB)));
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
