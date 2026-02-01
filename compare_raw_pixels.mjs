#!/usr/bin/env node
/**
 * Compare raw pixel values before normalization.
 */
import { readFile } from 'node:fs/promises';
import { loadImage } from './packages/node/dist/index.js';

const imagePath = process.argv[2] || './python_reference/validation/images/Screenshot_20260201_193653.png';

async function main() {
  console.log(`Loading image with Sharp: ${imagePath}`);
  const image = await loadImage(imagePath);

  console.log(`\nImage info:`);
  console.log(`  Size: ${image.width} x ${image.height}`);
  console.log(`  Channels: ${image.channels}`);
  console.log(`  Channel order: ${image.channelOrder}`);

  // Load Python's raw RGB image
  console.log(`\nLoading Python's raw RGB image...`);
  const pythonNpy = await readFile('debug_output/image_rgb_uint8.npy');

  // Simple .npy parser (skip header, assume uint8 data)
  const headerEnd = pythonNpy.indexOf(Buffer.from('\n')) + 1;
  let dataStart = headerEnd;

  // Find the actual data start (after dict header)
  for (let i = 6; i < pythonNpy.length - 1; i++) {
    if (pythonNpy[i] === 0x0A) {  // newline
      dataStart = i + 1;
      break;
    }
  }

  const pythonData = pythonNpy.slice(dataStart);
  console.log(`  Python data size: ${pythonData.length}`);
  console.log(`  JS data size: ${image.data.length}`);

  if (pythonData.length !== image.data.length) {
    console.log(`\n✗ Data sizes don't match!`);
    return 1;
  }

  // Compare first 100 pixels
  console.log(`\nFirst 10 pixels comparison (RGB):`);
  console.log(`  Pixel | Python RGB        | JS RGB            | Match`);
  console.log(`  ` + '-'.repeat(60));

  let mismatchCount = 0;
  for (let i = 0; i < 30; i += 3) {
    const pyR = pythonData[i];
    const pyG = pythonData[i + 1];
    const pyB = pythonData[i + 2];
    const jsR = image.data[i];
    const jsG = image.data[i + 1];
    const jsB = image.data[i + 2];

    const match = pyR === jsR && pyG === jsG && pyB === jsB;
    if (!match) mismatchCount++;

    const marker = match ? '✓' : '✗';
    console.log(`  ${(i/3).toString().padStart(5)} | [${pyR}, ${pyG}, ${pyB}]     | [${jsR}, ${jsG}, ${jsB}]     | ${marker}`);
  }

  // Check entire image
  let totalMismatches = 0;
  let maxDiff = 0;
  for (let i = 0; i < image.data.length; i++) {
    const diff = Math.abs(pythonData[i] - image.data[i]);
    if (diff > 0) totalMismatches++;
    if (diff > maxDiff) maxDiff = diff;
  }

  console.log(`\nFull image comparison:`);
  console.log(`  Total pixels: ${image.data.length / 3}`);
  console.log(`  Mismatched bytes: ${totalMismatches} (${(totalMismatches/image.data.length*100).toFixed(2)}%)`);
  console.log(`  Max difference: ${maxDiff}`);

  if (totalMismatches === 0) {
    console.log(`\n✓ Raw pixels MATCH perfectly!`);
    console.log(`  The issue must be in the preprocessing/normalization.`);
    return 0;
  } else {
    console.log(`\n✗ Raw pixels DON'T MATCH!`);
    console.log(`  Sharp and OpenCV are loading the image differently.`);
    return 1;
  }
}

main().then(code => process.exit(code)).catch(console.error);
