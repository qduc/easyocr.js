#!/usr/bin/env node
/**
 * Debug the full OCR pipeline with detailed logging.
 */
import { loadImage, loadDetectorModel, loadRecognizerModel, loadCharset, recognize } from './packages/node/dist/index.js';

const imagePath = process.argv[2] || './python_reference/validation/images/Screenshot_20260201_193653.png';

async function main() {
  console.log('='.repeat(60));
  console.log('FULL OCR PIPELINE DEBUG');
  console.log('='.repeat(60));

  // Load everything
  console.log('\n1. Loading resources...');
  const image = await loadImage(imagePath);
  console.log(`   Image: ${image.width}x${image.height}, ${image.channels} channels (${image.channelOrder})`);
  console.log(`   Data size: ${image.data.length} bytes (expected: ${image.width * image.height * image.channels})`);
  if (image.data.length !== image.width * image.height * image.channels) {
    console.log(`   ✗ WARNING: Data size mismatch!`);
  } else {
    console.log(`   ✓ Data size correct`);
  }
  console.log(`   First pixel: [${image.data[0]}, ${image.data[1]}, ${image.data[2]}]`);

  const detector = await loadDetectorModel('./models/onnx/craft_mlt_25k.onnx');
  console.log(`   ✓ Detector loaded`);

  const charset = await loadCharset('./models/english_g2.charset.txt');
  console.log(`   ✓ Charset loaded: ${charset.length} characters`);
  console.log(`     First 10 chars: "${charset.substring(0, 10)}"`);
  console.log(`     Last 10 chars: "${charset.substring(charset.length - 10)}"`);

  const recognizer = await loadRecognizerModel('./models/onnx/english_g2.onnx', {
    charset,
    textInputName: 'text',
  });
  console.log(`   ✓ Recognizer loaded`);

  // Run full pipeline
  console.log('\n2. Running full OCR pipeline...');
  const results = await recognize({ image, detector, recognizer });

  console.log(`\n3. Results: ${results.length} detections`);
  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    console.log(`   [${i}] "${result.text}" (confidence: ${result.confidence.toFixed(4)})`);
    const box = result.box;
    const boxStr = box.map(p => `(${Math.round(p[0])},${Math.round(p[1])})`).join(' ');
    console.log(`       Box: ${boxStr}`);
  }

  console.log('\n4. Expected results (from Python):');
  console.log('   [0] "V13.2" (confidence: 0.9277)');
  console.log('   [1] "30 May 2021" (confidence: 0.8519)');
  console.log('   [2] "Version 1,.3.2" (confidence: 0.5781)');

  console.log('\n5. Comparison:');
  const expected = ['V13.2', '30 May 2021', 'Version 1,.3.2'];
  for (let i = 0; i < Math.min(results.length, expected.length); i++) {
    const match = results[i].text === expected[i] ? '✓' : '✗';
    console.log(`   ${match} Expected: "${expected[i]}", Got: "${results[i].text}"`);
  }
}

main().catch(console.error);
