#!/usr/bin/env node
/**
 * Test to verify that JS preprocessing produces the same tensor as Python.
 */
import { readFile } from 'node:fs/promises';
import { loadImage } from './packages/node/dist/index.js';
import { detectorPreprocess, resolveOcrOptions } from './packages/core/dist/index.js';

const imagePath = process.argv[2] || './python_reference/validation/images/Screenshot_20260201_193653.png';
const pythonTensorPath = process.argv[3] || './debug_output/detector_input_python.npy';

async function loadNumpyArray(path) {
  const buffer = await readFile(path);

  // Parse .npy format (simplified - assumes float32, C-contiguous)
  // Magic: \x93NUMPY
  const magic = buffer.toString('ascii', 0, 6);
  if (magic !== '\x93NUMPY') {
    throw new Error('Not a valid .npy file');
  }

  // Version
  const major = buffer[6];
  const minor = buffer[7];

  // Header length (little-endian uint16 or uint32)
  let headerLen;
  if (major === 1) {
    headerLen = buffer.readUInt16LE(8);
    var dataOffset = 10 + headerLen;
  } else if (major === 2) {
    headerLen = buffer.readUInt32LE(8);
    var dataOffset = 12 + headerLen;
  } else {
    throw new Error(`Unsupported .npy version: ${major}.${minor}`);
  }

  // Parse header (Python dict string)
  const header = buffer.toString('ascii', major === 1 ? 10 : 12, dataOffset);

  // Extract shape
  const shapeMatch = header.match(/'shape':\s*\(([^)]+)\)/);
  if (!shapeMatch) throw new Error('Could not parse shape from .npy header');
  const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

  // Extract dtype
  const dtypeMatch = header.match(/'descr':\s*'([^']+)'/);
  if (!dtypeMatch) throw new Error('Could not parse dtype from .npy header');
  const dtype = dtypeMatch[1];

  if (dtype !== '<f4' && dtype !== '|f4') {
    throw new Error(`Expected float32 dtype, got: ${dtype}`);
  }

  // Read data
  const data = new Float32Array(buffer.buffer, buffer.byteOffset + dataOffset,
    (buffer.length - dataOffset) / 4);

  return { data, shape };
}

async function main() {
  console.log('='.repeat(60));
  console.log('TENSOR MATCH TEST');
  console.log('='.repeat(60));

  console.log(`\nLoading Python reference tensor from: ${pythonTensorPath}`);
  const pythonTensor = await loadNumpyArray(pythonTensorPath);
  console.log(`Python tensor shape: [${pythonTensor.shape.join(', ')}]`);
  console.log(`Python tensor data length: ${pythonTensor.data.length}`);

  console.log(`\nLoading and preprocessing image: ${imagePath}`);
  const image = await loadImage(imagePath);
  const options = resolveOcrOptions();
  const { input: jsTensor } = detectorPreprocess(image, options);

  console.log(`JS tensor shape: [${jsTensor.shape.join(', ')}]`);
  console.log(`JS tensor data length: ${jsTensor.data.length}`);

  // Compare shapes
  console.log('\n' + '='.repeat(60));
  console.log('SHAPE COMPARISON');
  console.log('='.repeat(60));

  const shapeMatch = pythonTensor.shape.length === jsTensor.shape.length &&
    pythonTensor.shape.every((dim, i) => dim === jsTensor.shape[i]);

  if (shapeMatch) {
    console.log('✓ Shapes MATCH: [' + pythonTensor.shape.join(', ') + ']');
  } else {
    console.log('✗ Shapes DO NOT MATCH:');
    console.log(`  Python: [${pythonTensor.shape.join(', ')}]`);
    console.log(`  JS:     [${jsTensor.shape.join(', ')}]`);
    console.log('\nTest FAILED: Shape mismatch');
    process.exit(1);
  }

  // Compare values
  console.log('\n' + '='.repeat(60));
  console.log('VALUE COMPARISON');
  console.log('='.repeat(60));

  const pyData = pythonTensor.data;
  const jsData = jsTensor.data;

  if (pyData.length !== jsData.length) {
    console.log(`✗ Data lengths differ: Python=${pyData.length}, JS=${jsData.length}`);
    process.exit(1);
  }

  // Calculate statistics
  let maxAbsDiff = 0;
  let sumAbsDiff = 0;
  let mismatchCount = 0;
  const tolerance = 1e-5;

  for (let i = 0; i < pyData.length; i++) {
    const diff = Math.abs(pyData[i] - jsData[i]);
    sumAbsDiff += diff;
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    if (diff > tolerance) mismatchCount++;
  }

  const meanAbsDiff = sumAbsDiff / pyData.length;

  console.log(`Total elements: ${pyData.length}`);
  console.log(`Max absolute difference: ${maxAbsDiff.toFixed(8)}`);
  console.log(`Mean absolute difference: ${meanAbsDiff.toFixed(8)}`);
  console.log(`Elements exceeding tolerance (${tolerance}): ${mismatchCount} (${(mismatchCount/pyData.length*100).toFixed(2)}%)`);

  // Show first 10 values for comparison
  console.log('\nFirst 10 values comparison:');
  console.log('  Index | Python      | JS          | Diff');
  console.log('  ' + '-'.repeat(50));
  for (let i = 0; i < Math.min(10, pyData.length); i++) {
    const diff = Math.abs(pyData[i] - jsData[i]);
    const marker = diff > tolerance ? ' ✗' : ' ✓';
    console.log(`  ${i.toString().padStart(5)} | ${pyData[i].toFixed(6)} | ${jsData[i].toFixed(6)} | ${diff.toFixed(8)}${marker}`);
  }

  // Overall result
  console.log('\n' + '='.repeat(60));
  console.log('RESULT');
  console.log('='.repeat(60));

  if (maxAbsDiff < tolerance) {
    console.log('✓ Test PASSED: Tensors match within tolerance!');
    console.log(`  All ${pyData.length} elements are within ${tolerance} of each other.`);
    return 0;
  } else {
    console.log(`✗ Test FAILED: Tensors differ by up to ${maxAbsDiff.toFixed(8)}`);
    console.log(`  ${mismatchCount} out of ${pyData.length} elements exceed tolerance.`);

    // Show some of the worst mismatches
    const mismatches = [];
    for (let i = 0; i < pyData.length; i++) {
      const diff = Math.abs(pyData[i] - jsData[i]);
      if (diff > tolerance) {
        mismatches.push({ index: i, py: pyData[i], js: jsData[i], diff });
      }
    }

    if (mismatches.length > 0) {
      mismatches.sort((a, b) => b.diff - a.diff);
      console.log(`\nTop 5 worst mismatches:`);
      console.log('  Index | Python      | JS          | Diff');
      console.log('  ' + '-'.repeat(50));
      for (let i = 0; i < Math.min(5, mismatches.length); i++) {
        const m = mismatches[i];
        console.log(`  ${m.index.toString().padStart(5)} | ${m.py.toFixed(6)} | ${m.js.toFixed(6)} | ${m.diff.toFixed(8)}`);
      }
    }

    return 1;
  }
}

main().then(code => process.exit(code)).catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
