#!/usr/bin/env node
/**
 * Debug script to inspect detector outputs in JS.
 */
import { loadImage, loadDetectorModel } from './packages/node/dist/index.js';
import { detectorPreprocess, resolveOcrOptions } from './packages/core/dist/index.js';

const imagePath = process.argv[2] || './python_reference/validation/images/Screenshot_20260201_193653.png';
const modelPath = './models/onnx/craft_mlt_25k.onnx';

async function main() {
  console.log('Loading image...');
  const image = await loadImage(imagePath);

  console.log('Loading detector model...');
  const detector = await loadDetectorModel(modelPath);

  console.log('\nPreprocessing...');
  const options = resolveOcrOptions();
  const { input } = detectorPreprocess(image, options);

  console.log(`Input tensor: shape=[${input.shape.join(', ')}], type=${input.type}`);

  console.log('\nRunning detector...');
  const outputs = await detector.session.run({ [detector.inputName]: input });

  console.log('\nDetector outputs:');
  for (const [name, tensor] of Object.entries(outputs)) {
    console.log(`  ${name}:`);
    console.log(`    Shape: [${tensor.shape.join(', ')}]`);
    console.log(`    Type: ${tensor.type}`);

    const data = tensor.data;
    let min = Infinity, max = -Infinity, sum = 0;
    for (let i = 0; i < data.length; i++) {
      const val = data[i];
      if (val < min) min = val;
      if (val > max) max = val;
      sum += val;
    }
    const mean = sum / data.length;

    console.log(`    Range: [${min.toFixed(6)}, ${max.toFixed(6)}]`);
    console.log(`    Mean: ${mean.toFixed(6)}`);

    // For the first output (should be text+link heatmap)
    if (tensor.shape.length === 4) {
      const [batch, h, w, channels] = tensor.shape;
      if (channels === 2) {
        // This is the combined text+link output
        let textAboveThreshold = 0;
        let linkAboveThreshold = 0;
        const textThreshold = 0.7;
        const linkThreshold = 0.4;

        // Iterate through the heatmap
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * 2;
            if (data[idx] > textThreshold) textAboveThreshold++;
            if (data[idx + 1] > linkThreshold) linkAboveThreshold++;
          }
        }

        console.log(`    Text pixels > ${textThreshold}: ${textAboveThreshold}`);
        console.log(`    Link pixels > ${linkThreshold}: ${linkAboveThreshold}`);
      }
    }
  }
}

main().catch(console.error);
