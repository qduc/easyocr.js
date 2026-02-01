#!/usr/bin/env node
/**
 * Debug script to dump intermediate tensors from JS implementation for comparison with Python.
 */
import { writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { loadImage, loadDetectorModel } from './packages/node/dist/index.js';
import { detectorPreprocess, resolveOcrOptions } from './packages/core/dist/index.js';

const imagePath = process.argv[2] || './python_reference/validation/images/Screenshot_20260201_193653.png';
const outputDir = process.argv[3] || './debug_output';

async function main() {
  console.log(`Loading image: ${imagePath}`);

  // Create output directory
  await mkdir(outputDir, { recursive: true });

  // Load image using sharp (same as production code)
  const image = await loadImage(imagePath);

  console.log(`\nImage info (loaded via sharp):`);
  console.log(`  Size: ${image.width} x ${image.height}`);
  console.log(`  Channels: ${image.channels}`);
  console.log(`  Channel order: ${image.channelOrder}`);
  console.log(`  Data type: ${image.data.constructor.name}`);
  console.log(`  Data length: ${image.data.length}`);
  console.log(`  First pixel (${image.channelOrder.toUpperCase()}): [${image.data[0]}, ${image.data[1]}, ${image.data[2]}]`);

  // Save raw image data
  await writeFile(
    `${outputDir}/image_${image.channelOrder}_uint8.json`,
    JSON.stringify({
      width: image.width,
      height: image.height,
      channels: image.channels,
      channelOrder: image.channelOrder,
      firstPixel: [image.data[0], image.data[1], image.data[2]],
      dataLength: image.data.length,
    }, null, 2)
  );

  // Apply detector preprocessing
  const options = resolveOcrOptions();

  console.log(`\nPreprocessing options:`);
  console.log(`  mean: [${options.mean.join(', ')}]`);
  console.log(`  std: [${options.std.join(', ')}]`);
  console.log(`  canvasSize: ${options.canvasSize}`);
  console.log(`  magRatio: ${options.magRatio}`);

  const { input: detectorInput, resized, scaleX, scaleY } = detectorPreprocess(image, options);

  console.log(`\nDetector preprocessing results:`);
  console.log(`  Input tensor shape: [${detectorInput.shape.join(', ')}]`);
  console.log(`  Input tensor type: ${detectorInput.type}`);
  console.log(`  Resized image: ${resized.width} x ${resized.height}`);
  console.log(`  Scale: ${scaleX.toFixed(6)} x ${scaleY.toFixed(6)}`);

  const data = detectorInput.data;
  let min = Infinity, max = -Infinity, sum = 0;
  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    if (val < min) min = val;
    if (val > max) max = val;
    sum += val;
  }
  const mean = sum / data.length;

  // Calculate std dev
  let variance = 0;
  for (let i = 0; i < data.length; i++) {
    variance += Math.pow(data[i] - mean, 2);
  }
  variance /= data.length;
  const std = Math.sqrt(variance);

  const stats = { min, max, mean, std };

  console.log(`\nDetector input tensor statistics:`);
  console.log(`  Range: [${stats.min.toFixed(6)}, ${stats.max.toFixed(6)}]`);
  console.log(`  Mean: ${stats.mean.toFixed(6)}`);
  console.log(`  Std: ${stats.std.toFixed(6)}`);
  console.log(`  First 10 values: [${Array.from(data.slice(0, 10)).map(v => v.toFixed(6)).join(', ')}]`);

  // Save detector input info
  await writeFile(
    `${outputDir}/detector_input_js.json`,
    JSON.stringify({
      shape: detectorInput.shape,
      type: detectorInput.type,
      min: stats.min,
      max: stats.max,
      mean: stats.mean,
      std: stats.std,
      first_10_values: Array.from(data.slice(0, 10)),
      first_100_values: Array.from(data.slice(0, 100)),
    }, null, 2)
  );

  // Save the full tensor as a binary file (float32 array)
  const buffer = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  await writeFile(`${outputDir}/detector_input_js.bin`, buffer);

  console.log(`\n✓ Saved JS tensors to ${outputDir}/`);
  console.log(`  - image_${image.channelOrder}_uint8.json (image metadata)`);
  console.log(`  - detector_input_js.json (tensor statistics)`);
  console.log(`  - detector_input_js.bin (raw float32 tensor data)`);

  console.log(`\nTo compare with Python:`);
  console.log(`  1. Run: python3 debug_tensors.py "${imagePath}" --output-dir "${outputDir}"`);
  console.log(`  2. Compare the JSON files for discrepancies`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
