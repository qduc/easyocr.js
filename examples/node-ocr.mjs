import { access, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { loadDetectorModel, loadImage, loadRecognizerModel, recognize } from '../packages/node/dist/index.js';

const repoRoot = new URL('..', import.meta.url);
const imagePath = process.argv[2]
  ? fileURLToPath(new URL(process.argv[2], `file://${process.cwd()}/`))
  : fileURLToPath(new URL('./Screenshot_20260201_193653.png', import.meta.url));

const detectorPath = fileURLToPath(new URL('./models/onnx/craft_mlt_25k.onnx', repoRoot));
const recognizerPath = fileURLToPath(new URL('./models/onnx/english_g2.onnx', repoRoot));
const charsetPath = fileURLToPath(new URL('./models/english_g2.charset.txt', repoRoot));

const ensureFile = async (path, label) => {
  try {
    await access(path);
  } catch {
    throw new Error(`${label} not found at ${path}`);
  }
};

const main = async () => {
  await ensureFile(imagePath, 'Image');
  await ensureFile(detectorPath, 'Detector ONNX');
  await ensureFile(recognizerPath, 'Recognizer ONNX');
  await ensureFile(charsetPath, 'Charset file');

  const charset = (await readFile(charsetPath, 'utf8')).trimEnd();
  if (!charset) {
    throw new Error(`Charset file is empty: ${charsetPath}`);
  }

  const image = await loadImage(imagePath);
  const detector = await loadDetectorModel(detectorPath);
  const recognizer = await loadRecognizerModel(recognizerPath, {
    charset,
    textInputName: 'text',
  });
  const results = await recognize({ image, detector, recognizer });
  console.log(JSON.stringify(results, null, 2));
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
