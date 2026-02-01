import { readFile } from 'node:fs/promises';
import {
  cropHorizontal,
  ctcGreedyDecode,
  loadImage,
  loadRecognizerModel,
  recognizerPreprocess,
  resolveOcrOptions,
} from '../packages/node/dist/index.js';

const imagePath = './python_reference/validation/images/Screenshot_20260201_193653.png';
const recognizerPath = './models/onnx/english_g2.onnx';
const charsetPath = './models/english_g2.charset.txt';

const box = [
  [6, 18],
  [104, 18],
  [104, 50],
  [6, 50],
];

const main = async () => {
  const charset = (await readFile(charsetPath, 'utf8')).trimEnd();
  const image = await loadImage(imagePath);
  const recognizer = await loadRecognizerModel(recognizerPath, {
    charset,
    textInputName: 'text',
  });
  const [crop] = cropHorizontal(image, [box]);
  const options = resolveOcrOptions();
  const prep = recognizerPreprocess(crop.image, options);
  const rawOptions = { ...options, recognizer: { ...options.recognizer, mean: 0, std: 1 } };
  const rawPrep = recognizerPreprocess(crop.image, rawOptions);
  const feeds = { [recognizer.inputName]: prep.input };
  if (recognizer.textInputName) {
    feeds[recognizer.textInputName] = {
      data: new Int32Array([0]),
      shape: [1, 1],
      type: 'int32',
    };
  }
  const outputs = await recognizer.session.run(feeds);
  const data = prep.input.data;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  console.log('input stats', { min, max, mean: sum / data.length });
  const logits = outputs[recognizer.outputName];
  const steps = logits.shape[logits.shape.length - 2];
  const classes = logits.shape[logits.shape.length - 1];
  const candidates = [recognizer.blankIndex ?? 0, 0, classes - 1, 1];
  console.log('normal');
  for (const blankIndex of candidates) {
    const decoded = ctcGreedyDecode(logits.data, steps, classes, recognizer.charset, blankIndex);
    console.log(`blankIndex=${blankIndex}`, decoded);
  }
  const argmax = [];
  for (let t = 0; t < steps; t += 1) {
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < classes; c += 1) {
      const v = logits.data[t * classes + c];
      if (v > bestScore) {
        bestScore = v;
        bestIdx = c;
      }
    }
    argmax.push(bestIdx);
  }
  console.log('argmax', argmax);

  const rawFeeds = { ...feeds, [recognizer.inputName]: rawPrep.input };
  const rawOutputs = await recognizer.session.run(rawFeeds);
  const rawLogits = rawOutputs[recognizer.outputName];
  console.log('no-normalize');
  for (const blankIndex of candidates) {
    const decoded = ctcGreedyDecode(rawLogits.data, steps, classes, recognizer.charset, blankIndex);
    console.log(`blankIndex=${blankIndex}`, decoded);
  }

  const inverted = new Float32Array(prep.input.data.length);
  for (let i = 0; i < prep.input.data.length; i += 1) {
    inverted[i] = -prep.input.data[i];
  }
  const invFeeds = { ...feeds, [recognizer.inputName]: { ...prep.input, data: inverted } };
  const invOutputs = await recognizer.session.run(invFeeds);
  const invLogits = invOutputs[recognizer.outputName];
  console.log('inverted');
  for (const blankIndex of candidates) {
    const decoded = ctcGreedyDecode(invLogits.data, steps, classes, recognizer.charset, blankIndex);
    console.log(`blankIndex=${blankIndex}`, decoded);
  }
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
