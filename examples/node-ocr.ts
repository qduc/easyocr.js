import { access, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { guessModel, loadDetectorModel, loadImage, loadRecognizerModel, recognize } from '@qduc/easyocr-node';

const repoRoot = new URL('..', import.meta.url);
const args = process.argv.slice(2);
let model = 'english_g2';
let langList: string[] | undefined = undefined;
let allowlist: string | undefined = undefined;
let blocklist: string | undefined = undefined;
let imagePathArg = '';

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--model' && args[i + 1]) {
    model = args[i + 1];
    i++;
  } else if (args[i] === '--langs' && args[i + 1]) {
    langList = args[i + 1].split(',');
    i++;
  } else if (args[i] === '--allowlist' && args[i + 1]) {
    allowlist = args[i + 1];
    i++;
  } else if (args[i] === '--blocklist' && args[i + 1]) {
    blocklist = args[i + 1];
    i++;
  } else if (!args[i].startsWith('--')) {
    imagePathArg = args[i];
  }
}

const imagePath = imagePathArg
  ? fileURLToPath(new URL(imagePathArg, `file://${process.cwd()}/`))
  : fileURLToPath(new URL('./Screenshot_20260201_193653.png', import.meta.url));

let finalModel = model;
if (!args.includes('--model') && langList && langList.length > 0) {
  finalModel = guessModel(langList);
  console.log(`Guessed model "${finalModel}" from languages: ${langList.join(', ')}`);
}

const detectorPath = fileURLToPath(new URL('./models/onnx/craft_mlt_25k.onnx', repoRoot));
const recognizerPath = fileURLToPath(new URL(`./models/onnx/${finalModel}.onnx`, repoRoot));
const charsetPath = fileURLToPath(new URL(`./models/${finalModel}.charset.txt`, repoRoot));

const ensureFile = async (path: string, label: string) => {
  try {
    await access(path);
  } catch {
    throw new Error(`${label} not found at ${path}. Run \`python models/export_onnx.py --detector --recognizer\` to export models.`);
  }
};

const main = async () => {
  await ensureFile(imagePath, 'Image');
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
  const results = await recognize({
    image,
    detector,
    recognizer,
    options: {
      langList,
      allowlist,
      blocklist,
    },
  });
  console.log(JSON.stringify(results, null, 2));
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
