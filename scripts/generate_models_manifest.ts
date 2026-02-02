import { createHash } from 'node:crypto';
import { createReadStream, promises as fs } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

type ModelKind = 'detector' | 'recognizer';

interface ModelManifestEntry {
  modelName: string;
  kind: ModelKind;
  languages: string[];
  charsetFile?: string;
  onnxFile: string;
  textInputName?: string;
  sha256: string;
  size: number;
}

interface ModelsManifest {
  schemaVersion: 1;
  package: '@qduc/easyocr-core';
  packageVersion: string;
  generatedAt: string;
  models: ModelManifestEntry[];
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const sha256File = async (filePath: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
};

const main = async () => {
  const corePkgPath = path.join(repoRoot, 'packages/core/package.json');
  const corePkg = JSON.parse(await fs.readFile(corePkgPath, 'utf8')) as { version: string };

  const modelRoot = path.join(repoRoot, 'models');
  const onnxDir = path.join(modelRoot, 'onnx');

  const entries: Array<Omit<ModelManifestEntry, 'sha256' | 'size'>> = [
    {
      modelName: 'craft_mlt_25k',
      kind: 'detector',
      languages: ['*'],
      onnxFile: 'onnx/craft_mlt_25k.onnx',
    },
    {
      modelName: 'english_g2',
      kind: 'recognizer',
      languages: ['en'],
      charsetFile: 'english_g2.charset.txt',
      onnxFile: 'onnx/english_g2.onnx',
    },
    {
      modelName: 'latin_g2',
      kind: 'recognizer',
      languages: [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga',
        'hr', 'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt',
        'nl', 'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq',
        'sv', 'sw', 'tl', 'tr', 'uz', 'vi',
      ],
      charsetFile: 'latin_g2.charset.txt',
      onnxFile: 'onnx/latin_g2.onnx',
    },
    {
      modelName: 'cyrillic_g2',
      kind: 'recognizer',
      languages: [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd',
        'ava', 'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'tjk',
      ],
      charsetFile: 'cyrillic_g2.charset.txt',
      onnxFile: 'onnx/cyrillic_g2.onnx',
    },
    {
      modelName: 'japanese_g2',
      kind: 'recognizer',
      languages: ['ja'],
      charsetFile: 'japanese_g2.charset.txt',
      onnxFile: 'onnx/japanese_g2.onnx',
    },
    {
      modelName: 'korean_g2',
      kind: 'recognizer',
      languages: ['ko'],
      charsetFile: 'korean_g2.charset.txt',
      onnxFile: 'onnx/korean_g2.onnx',
    },
    {
      modelName: 'zh_sim_g2',
      kind: 'recognizer',
      languages: ['ch_sim'],
      charsetFile: 'zh_sim_g2.charset.txt',
      onnxFile: 'onnx/zh_sim_g2.onnx',
    },
    {
      modelName: 'telugu_g2',
      kind: 'recognizer',
      languages: ['te'],
      charsetFile: 'telugu_g2.charset.txt',
      onnxFile: 'onnx/telugu_g2.onnx',
    },
    {
      modelName: 'kannada_g2',
      kind: 'recognizer',
      languages: ['kn'],
      charsetFile: 'kannada_g2.charset.txt',
      onnxFile: 'onnx/kannada_g2.onnx',
    },
  ];

  const models: ModelManifestEntry[] = [];
  for (const entry of entries) {
    const absOnnxPath = path.join(modelRoot, entry.onnxFile);
    const stat = await fs.stat(absOnnxPath);
    const sha256 = await sha256File(absOnnxPath);
    models.push({
      ...entry,
      size: stat.size,
      sha256,
    });
  }

  const manifest: ModelsManifest = {
    schemaVersion: 1,
    package: '@qduc/easyocr-core',
    packageVersion: corePkg.version,
    generatedAt: new Date().toISOString(),
    models,
  };

  const outJson = JSON.stringify(manifest, null, 2) + '\n';

  const rootOutPath = path.join(modelRoot, 'manifest.json');
  await fs.writeFile(rootOutPath, outJson, 'utf8');

  const pkgOutDir = path.join(repoRoot, 'packages/core/models');
  await fs.mkdir(pkgOutDir, { recursive: true });
  const pkgOutPath = path.join(pkgOutDir, 'manifest.json');
  await fs.writeFile(pkgOutPath, outJson, 'utf8');

  console.log(`Wrote ${rootOutPath}`);
  console.log(`Wrote ${pkgOutPath}`);

  // Basic sanity checks
  for (const m of manifest.models) {
    const abs = path.join(modelRoot, m.onnxFile);
    if (!abs.startsWith(onnxDir + path.sep)) {
      throw new Error(`onnxFile must be under models/onnx/: ${m.onnxFile}`);
    }
    if (m.kind === 'recognizer' && !m.charsetFile) {
      throw new Error(`recognizer missing charsetFile: ${m.modelName}`);
    }
  }
};

await main();
