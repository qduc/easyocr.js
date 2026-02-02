import { access, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import * as easyocr from '@qduc/easyocr-node';

const repoRoot = new URL('..', import.meta.url);
const argv = process.argv.slice(2);
let imageArg = null;
let traceDir = null;
let langs = [];
let useGpu = false;
for (let i = 0; i < argv.length; i += 1) {
  const arg = argv[i];
  if (arg === '--trace-dir') {
    traceDir = argv[i + 1] ?? null;
    i += 1;
    continue;
  }
  if (arg === '--langs' || arg === '--lang') {
    const value = argv[i + 1] ?? '';
    langs = value.split(',').map((lang) => lang.trim()).filter(Boolean);
    i += 1;
    continue;
  }
  if (arg === '--gpu') {
    useGpu = true;
    continue;
  }
  if (!arg.startsWith('-') && imageArg === null) {
    imageArg = arg;
  }
}
if (langs.length === 0) {
  langs = ['en'];
}

const imagePath = imageArg
  ? fileURLToPath(new URL(imageArg, `file://${process.cwd()}/`))
  : fileURLToPath(new URL('./Screenshot_20260201_193653.png', import.meta.url));

const detectorPath = fileURLToPath(new URL('./models/onnx/craft_mlt_25k.onnx', repoRoot));
const recognizerModel = easyocr.guessModel(langs);
const recognizerPath = fileURLToPath(new URL(`./models/onnx/${recognizerModel}.onnx`, repoRoot));
const charsetPath = fileURLToPath(new URL(`./models/${recognizerModel}.charset.txt`, repoRoot));

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

  const image = await easyocr.loadImage(imagePath);
  const recognitionImage =
    typeof easyocr.loadGrayscaleImage === 'function' ? await easyocr.loadGrayscaleImage(imagePath) : undefined;
  const sessionOptions = useGpu ? { executionProviders: ['cuda', 'coreml', 'cpu'] } : undefined;
  const detector = await easyocr.loadDetectorModel(detectorPath, { sessionOptions });
  const recognizer = await easyocr.loadRecognizerModel(recognizerPath, {
    charset,
    textInputName: 'text',
    sessionOptions,
  });
  const trace =
    traceDir && typeof easyocr.createFsTraceWriter === 'function'
      ? easyocr.createFsTraceWriter({
          traceDir,
          runMeta: {
            impl: 'js',
            imagePath,
            detectorPath,
            recognizerPath,
            langs,
          },
        })
      : undefined;
  if (traceDir && !trace) {
    console.warn('Tracing requested but createFsTraceWriter is not available (run `bun run build` first).');
  }
  const results = await easyocr.recognize({
    image,
    recognitionImage,
    detector,
    recognizer,
    trace,
    options: {
      langList: langs,
    },
  });
  console.log(JSON.stringify(results, null, 2));
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
