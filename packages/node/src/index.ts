export * from '@qduc/easyocr-core';
export * from './trace.js';

import { guessModel, recognize } from '@qduc/easyocr-core';
import type {
  DetectorModel,
  InferenceRunner,
  OcrOptions,
  OcrResult,
  RasterImage,
  RecognizerModel,
  Tensor,
  TensorType,
} from '@qduc/easyocr-core';
import { access, mkdir, writeFile, readFile } from 'node:fs/promises';
import * as nodePath from 'node:path';
import sharp from 'sharp';
import * as ort from 'onnxruntime-node';

const MODELS_BASE_URL = 'https://github.com/qduc/easyocr.js/releases/download/v0.0.1-alpha';

async function ensureModel(modelPath: string): Promise<void> {
  try {
    await access(modelPath);
  } catch {
    const filename = nodePath.basename(modelPath);
    // Only attempt to download if it looks like an ONNX model we know about
    if (!filename.endsWith('.onnx')) {
      return;
    }

    const url = `${MODELS_BASE_URL}/${filename}`;
    console.log(`Model not found at ${modelPath}. Downloading from ${url}...`);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to download model from ${url}: ${response.status} ${response.statusText}. ` +
        `Please ensure the model exists or run the export script manually.`,
      );
    }

    const buffer = await response.arrayBuffer();
    await mkdir(nodePath.dirname(modelPath), { recursive: true });
    await writeFile(modelPath, new Uint8Array(buffer));
    console.log(`Model successfully downloaded to ${modelPath}`);
  }
}

export type NodeImageInput = string | Buffer | Uint8Array;

export interface LoadImageOptions {
  channelOrder?: 'rgb';
}

const inferPath = (input: NodeImageInput): string | null => (typeof input === 'string' ? input : null);

export const loadImage = async (
  input: NodeImageInput,
  options: LoadImageOptions = {},
): Promise<RasterImage> => {
  const channelOrder = options.channelOrder ?? 'rgb';
  const image =
    typeof input === 'string'
      ? sharp(input).toColourspace('srgb').removeAlpha().raw()
      : sharp(Buffer.from(input instanceof Uint8Array ? input : new Uint8Array(input))).toColourspace('srgb').removeAlpha().raw();
  const { data, info } = await image.toBuffer({ resolveWithObject: true });

  // After removeAlpha() and toColourspace('srgb'), we should have 3 channels (RGB)
  // But Sharp might return fewer channels for grayscale images
  if (info.channels !== 3) {
    throw new Error(`Expected 3 channels after RGB conversion, got ${info.channels}`);
  }

  return {
    data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    width: info.width,
    height: info.height,
    channels: 3,
    channelOrder,
  };
};

export const loadGrayscaleImage = async (input: NodeImageInput): Promise<RasterImage> => {
  const image =
    typeof input === 'string'
      ? sharp(input).toColourspace('b-w').removeAlpha().raw()
      : sharp(Buffer.from(input instanceof Uint8Array ? input : new Uint8Array(input)))
          .toColourspace('b-w')
          .removeAlpha()
          .raw();
  const { data, info } = await image.toBuffer({ resolveWithObject: true });
  if (info.channels !== 1) {
    throw new Error(`Expected 1 channel for grayscale image, got ${info.channels}`);
  }
  return {
    data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    width: info.width,
    height: info.height,
    channels: 1,
    channelOrder: 'gray',
  };
};

const toOrtType = (type: TensorType): ort.Tensor.Type => {
  switch (type) {
    case 'float32':
      return 'float32';
    case 'int32':
      return 'int32';
    case 'uint8':
      return 'uint8';
    default:
      return 'float32';
  }
};

const fromOrtType = (type: ort.Tensor.Type): TensorType => {
  if (type === 'float32' || type === 'int32' || type === 'uint8') {
    return type;
  }
  throw new Error(`Unsupported tensor type: ${type}`);
};

const toOrtTensor = (tensor: Tensor): ort.Tensor =>
  new ort.Tensor(toOrtType(tensor.type), tensor.data, tensor.shape);

const fromOrtTensor = (tensor: ort.Tensor): Tensor => ({
  data: tensor.data as Float32Array | Int32Array | Uint8Array,
  shape: [...tensor.dims],
  type: fromOrtType(tensor.type),
});

class OrtRunner implements InferenceRunner {
  protected session: ort.InferenceSession;

  constructor(session: ort.InferenceSession) {
    this.session = session;
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const ortFeeds: Record<string, ort.Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      ortFeeds[name] = toOrtTensor(tensor);
    }
    const outputs = await this.session.run(ortFeeds);
    const result: Record<string, Tensor> = {};
    for (const [name, tensor] of Object.entries(outputs)) {
      result[name] = fromOrtTensor(tensor);
    }
    return result;
  }
}

const splitCraftTensor = (tensor: Tensor): { text: Tensor; link: Tensor } => {
  if (tensor.type !== 'float32') {
    throw new Error('Detector output must be float32.');
  }
  const dims = tensor.shape;
  if (dims.length !== 4) {
    throw new Error(`Detector output must be 4D, got shape ${dims.join('x')}`);
  }
  const data = tensor.data as Float32Array;
  if (dims[3] === 2) {
    const height = dims[1];
    const width = dims[2];
    const text = new Float32Array(width * height);
    const link = new Float32Array(width * height);
    let offset = 0;
    for (let i = 0; i < width * height; i += 1) {
      text[i] = data[offset];
      link[i] = data[offset + 1];
      offset += 2;
    }
    return {
      text: { data: text, shape: [1, height, width], type: 'float32' },
      link: { data: link, shape: [1, height, width], type: 'float32' },
    };
  }
  if (dims[1] === 2) {
    const height = dims[2];
    const width = dims[3];
    const channelStride = width * height;
    const text = data.subarray(0, channelStride);
    const link = data.subarray(channelStride, channelStride * 2);
    return {
      text: { data: text, shape: [1, height, width], type: 'float32' },
      link: { data: link, shape: [1, height, width], type: 'float32' },
    };
  }
  throw new Error(`Detector output channel dim not found in shape ${dims.join('x')}`);
};

class CraftRunner extends OrtRunner {
  private textName: string;
  private linkName: string;

  constructor(session: ort.InferenceSession, textName: string, linkName: string) {
    super(session);
    this.textName = textName;
    this.linkName = linkName;
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const ortFeeds: Record<string, ort.Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      ortFeeds[name] = toOrtTensor(tensor);
    }
    const outputs = await this.session.run(ortFeeds);
    if (outputs[this.textName] && outputs[this.linkName]) {
      return {
        [this.textName]: fromOrtTensor(outputs[this.textName]),
        [this.linkName]: fromOrtTensor(outputs[this.linkName]),
      };
    }
    const candidate = Object.values(outputs).find((tensor) => {
      const dims = tensor.dims;
      return dims.length === 4 && (dims[1] === 2 || dims[3] === 2);
    });
    if (!candidate) {
      throw new Error('Detector outputs do not include a 2-channel heatmap tensor.');
    }
    const split = splitCraftTensor(fromOrtTensor(candidate));
    return {
      [this.textName]: split.text,
      [this.linkName]: split.link,
    };
  }
}

class RecognizerRunner extends OrtRunner {
  private textInputName?: string;

  constructor(session: ort.InferenceSession, textInputName?: string) {
    super(session);
    this.textInputName = textInputName;
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const ortFeeds: Record<string, ort.Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      if (this.textInputName && name === this.textInputName) {
        const src = tensor.data as Float32Array | Int32Array | Uint8Array;
        const big = new BigInt64Array(src.length);
        for (let i = 0; i < src.length; i += 1) {
          big[i] = BigInt(src[i]);
        }
        ortFeeds[name] = new ort.Tensor('int64', big, tensor.shape);
      } else {
        ortFeeds[name] = toOrtTensor(tensor);
      }
    }
    const outputs = await this.session.run(ortFeeds);
    const result: Record<string, Tensor> = {};
    for (const [name, tensor] of Object.entries(outputs)) {
      result[name] = fromOrtTensor(tensor);
    }
    return result;
  }
}

export interface LoadSessionOptions {
  sessionOptions?: ort.InferenceSession.SessionOptions;
}

export const loadSession = async (
  model: string | Uint8Array | Buffer,
  options: LoadSessionOptions = {},
): Promise<ort.InferenceSession> => {
  if (typeof model === 'string') {
    await ensureModel(model);
    return ort.InferenceSession.create(model, options.sessionOptions);
  }
  const data = model instanceof Uint8Array ? model : new Uint8Array(model);
  return ort.InferenceSession.create(data, options.sessionOptions);
};

export const loadCharset = async (path: string): Promise<string> => {
  const data = await readFile(path, 'utf8');
  return data.trimEnd();
};

export interface DetectorModelOptions extends LoadSessionOptions {
  inputName?: string;
  textOutputName?: string;
  linkOutputName?: string;
}

export const loadDetectorModel = async (
  model: string | Uint8Array | Buffer,
  options: DetectorModelOptions = {},
): Promise<DetectorModel> => {
  const session = await loadSession(model, options);
  const inputName = options.inputName ?? session.inputNames[0];
  const textOutputName = options.textOutputName ?? 'text';
  const linkOutputName = options.linkOutputName ?? 'link';
  if (!inputName || !textOutputName || !linkOutputName) {
    throw new Error('Detector model is missing required input/output names.');
  }
  return {
    session: new CraftRunner(session, textOutputName, linkOutputName),
    inputName,
    textOutputName,
    linkOutputName,
  };
};

export interface RecognizerModelOptions extends LoadSessionOptions {
  inputName?: string;
  outputName?: string;
  textInputName?: string;
  charset: string;
  symbols?: string;
  blankIndex?: number;
}

const resolveRecognizerTextInputName = (
  session: ort.InferenceSession,
  imageInputName: string,
  configured?: string,
): string | undefined => {
  const inputNames = session.inputNames ?? [];
  const others = inputNames.filter((n) => n && n !== imageInputName);

  if (configured) {
    if (configured !== imageInputName && inputNames.includes(configured)) return configured;
    const ci = others.find((n) => n.toLowerCase() === configured.toLowerCase());
    if (ci) return ci;
  }

  if (others.length === 1) return others[0];

  const preferred = ['text', 'text_input', 'textinput', 'input_text', 'tokens'];
  for (const candidate of preferred) {
    const hit = others.find((n) => n.toLowerCase() === candidate);
    if (hit) return hit;
  }

  return undefined;
};

export const loadRecognizerModel = async (
  model: string | Uint8Array | Buffer,
  options: RecognizerModelOptions,
): Promise<RecognizerModel> => {
  const session = await loadSession(model, options);
  const inputName = options.inputName ?? session.inputNames[0];
  const outputName = options.outputName ?? session.outputNames[0];
  if (!inputName || !outputName) {
    throw new Error('Recognizer model is missing required input/output names.');
  }
  if (!options.charset) {
    throw new Error('Recognizer charset is required.');
  }

  const textInputName = resolveRecognizerTextInputName(session, inputName, options.textInputName);
  return {
    session: new RecognizerRunner(session, textInputName),
    inputName,
    outputName,
    textInputName,
    charset: options.charset,
    symbols: options.symbols,
    blankIndex: options.blankIndex,
  };
};

export type OCRImageInput = NodeImageInput | RasterImage;

export interface OCRInstance {
  read(image: OCRImageInput, options?: Partial<OcrOptions>): Promise<OcrResult[]>;
}

export interface CreateOCROptions {
  modelDir: string;
  lang?: string;
  langs?: string[];
  detector?: DetectorModel;
  recognizer?: RecognizerModel;
  detectorModelPath?: string;
  recognizerModelPath?: string;
  charsetPath?: string;
  detectorOptions?: DetectorModelOptions;
  recognizerOptions?: Omit<RecognizerModelOptions, 'charset'>;
  sessionOptions?: LoadSessionOptions['sessionOptions'];
  ocrOptions?: Partial<OcrOptions>;
}

const mergeOcrOptions = (
  base?: Partial<OcrOptions>,
  override?: Partial<OcrOptions>,
): Partial<OcrOptions> | undefined => {
  if (!base && !override) return undefined;
  const merged: Partial<OcrOptions> = {
    ...base,
    ...override,
  };
  if (base?.recognizer && override?.recognizer) {
    merged.recognizer = { ...base.recognizer, ...override.recognizer };
  } else if (base?.recognizer) {
    merged.recognizer = base.recognizer;
  } else if (override?.recognizer) {
    merged.recognizer = override.recognizer;
  }
  return merged;
};

const normalizeLangs = (lang?: string, langs?: string[]): string[] => {
  const normalized = [...(langs ?? [])];
  if (lang && !normalized.includes(lang)) {
    normalized.unshift(lang);
  }
  return normalized.length > 0 ? normalized : ['en'];
};

const isRasterImage = (input: OCRImageInput): input is RasterImage => {
  if (!input || typeof input !== 'object') return false;
  const candidate = input as RasterImage;
  return (
    candidate.data instanceof Uint8Array &&
    typeof candidate.width === 'number' &&
    typeof candidate.height === 'number' &&
    typeof candidate.channels === 'number' &&
    typeof candidate.channelOrder === 'string'
  );
};

export const createOCR = async (options: CreateOCROptions): Promise<OCRInstance> => {
  if (!options?.modelDir) {
    throw new Error('createOCR requires a modelDir option.');
  }

  const langs = normalizeLangs(options.lang, options.langs);
  const recognizerName = guessModel(langs);
  const detectorModelPath =
    options.detectorModelPath ??
    nodePath.join(options.modelDir, 'onnx', 'craft_mlt_25k.onnx');
  const recognizerModelPath =
    options.recognizerModelPath ??
    nodePath.join(options.modelDir, 'onnx', `${recognizerName}.onnx`);
  const charsetPath =
    options.charsetPath ??
    nodePath.join(options.modelDir, `${recognizerName}.charset.txt`);

  const sessionOptions = options.sessionOptions;
  const detectorOptions = { sessionOptions, ...(options.detectorOptions ?? {}) };
  const recognizerOptions = { sessionOptions, ...(options.recognizerOptions ?? {}) };

  const detector =
    options.detector ?? (await loadDetectorModel(detectorModelPath, detectorOptions));
  const recognizer =
    options.recognizer ??
    (await (async () => {
      const charset = await loadCharset(charsetPath);
      return loadRecognizerModel(recognizerModelPath, {
        ...recognizerOptions,
        charset,
      });
    })());

  const baseOcrOptions: Partial<OcrOptions> = {
    ...(options.ocrOptions ?? {}),
  };
  if (!baseOcrOptions.langList) {
    baseOcrOptions.langList = langs;
  }

  return {
    async read(image: OCRImageInput, overrideOptions?: Partial<OcrOptions>) {
      const raster = isRasterImage(image) ? image : await loadImage(image);
      const mergedOptions = mergeOcrOptions(baseOcrOptions, overrideOptions);
      return recognize({
        image: raster,
        detector,
        recognizer,
        options: mergedOptions,
      });
    },
  };
};
