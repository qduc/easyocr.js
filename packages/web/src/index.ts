export * from '@qduc/easyocr-core';

import { version as coreVersion } from '@qduc/easyocr-core';
import type {
  DetectorModel,
  InferenceRunner,
  RasterImage,
  RecognizerModel,
  Tensor,
  TensorType,
} from '@qduc/easyocr-core';
import type * as ort from 'onnxruntime-web';

type OrtModule = typeof import('onnxruntime-web');

let ortModule: OrtModule | null = null;
const loadOrt = async (): Promise<OrtModule> => {
  if (!ortModule) {
    ortModule = await import('onnxruntime-web');
  }
  return ortModule;
};

export interface DefaultModelBaseUrlOptions {
  /** Git ref (tag/branch/SHA) in qduc/easyocr.js. Defaults to v${version} of @qduc/easyocr-core. */
  ref?: string;
}

export const getDefaultModelBaseUrl = (options: DefaultModelBaseUrlOptions = {}): string => {
  // Versioned by @qduc/easyocr-core so consumers can rely on stability.
  // Uses media.githubusercontent.com to avoid Git LFS pointer responses.
  const ref = options.ref ?? `v${coreVersion}`;
  return `https://media.githubusercontent.com/media/qduc/easyocr.js/${ref}/models`;
};

const isGitLfsPointer = (text: string): boolean => {
  const head = text.slice(0, 200);
  return head.includes('version https://git-lfs.github.com/spec/v1') && head.includes('oid sha256:');
};

export interface FetchModelOptions {
  signal?: AbortSignal;
}

export const fetchModel = async (url: string, options: FetchModelOptions = {}): Promise<Uint8Array> => {
  if (typeof fetch === 'undefined') {
    throw new Error('fetch is unavailable in this environment.');
  }

  const response = await fetch(url, { signal: options.signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch model from ${url}: ${response.status} ${response.statusText}`);
  }

  const contentType = response.headers.get('content-type') ?? '';
  const contentLength = Number(response.headers.get('content-length') ?? '0');

  // GitHub raw URLs for LFS objects frequently return a small text/plain pointer file.
  if (contentType.startsWith('text/') || (contentLength > 0 && contentLength < 2048)) {
    const text = await response.text();
    if (isGitLfsPointer(text)) {
      throw new Error(
        'You fetched a Git LFS pointer file, not the actual model. ' +
          'Use media.githubusercontent.com (or a release asset with CORS) instead of raw.githubusercontent.com.',
      );
    }
    return new TextEncoder().encode(text);
  }

  const buffer = await response.arrayBuffer();
  return new Uint8Array(buffer);
};

export type WebImageInput =
  | Blob
  | File
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas;

export interface LoadImageOptions {
  channelOrder?: 'rgb';
}

const isImageData = (input: unknown): input is ImageData =>
  typeof ImageData !== 'undefined' && input instanceof ImageData;
const isHtmlImage = (input: unknown): input is HTMLImageElement =>
  typeof HTMLImageElement !== 'undefined' && input instanceof HTMLImageElement;
const isHtmlCanvas = (input: unknown): input is HTMLCanvasElement =>
  typeof HTMLCanvasElement !== 'undefined' && input instanceof HTMLCanvasElement;
const isOffscreenCanvas = (input: unknown): input is OffscreenCanvas =>
  typeof OffscreenCanvas !== 'undefined' && input instanceof OffscreenCanvas;
const isBlob = (input: unknown): input is Blob =>
  typeof Blob !== 'undefined' && input instanceof Blob;
const isImageBitmap = (input: unknown): input is ImageBitmap =>
  typeof ImageBitmap !== 'undefined' && input instanceof ImageBitmap;

const ensureImageDecoded = async (image: HTMLImageElement): Promise<void> => {
  if (image.decode) {
    try {
      await image.decode();
      return;
    } catch {
      // Fall back to load events.
    }
  }
  if (image.complete && image.naturalWidth > 0) {
    return;
  }
  await new Promise<void>((resolve, reject) => {
    const onLoad = () => resolve();
    const onError = () => reject(new Error('Failed to load image element.'));
    image.addEventListener('load', onLoad, { once: true });
    image.addEventListener('error', onError, { once: true });
  });
};

const getCanvasContext = (
  canvas: HTMLCanvasElement | OffscreenCanvas,
): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D => {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas 2D context is unavailable.');
  }
  return ctx;
};

const createCanvas = (
  width: number,
  height: number,
): { canvas: HTMLCanvasElement | OffscreenCanvas; ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D } => {
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = getCanvasContext(canvas);
    return { canvas, ctx };
  }
  if (typeof document !== 'undefined' && document.createElement) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = getCanvasContext(canvas);
    return { canvas, ctx };
  }
  throw new Error('Canvas APIs are unavailable. Provide ImageData directly or run in a browser environment.');
};

const imageDataFromCanvas = (canvas: HTMLCanvasElement | OffscreenCanvas): ImageData => {
  const ctx = getCanvasContext(canvas);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
};

const imageDataFromSource = (
  source: CanvasImageSource,
  width: number,
  height: number,
): ImageData => {
  const { ctx } = createCanvas(width, height);
  ctx.drawImage(source, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
};

const imageDataFromBlob = async (blob: Blob): Promise<ImageData> => {
  if (typeof createImageBitmap !== 'undefined') {
    const bitmap = await createImageBitmap(blob);
    const data = imageDataFromSource(bitmap, bitmap.width, bitmap.height);
    if (typeof bitmap.close === 'function') {
      bitmap.close();
    }
    return data;
  }
  if (typeof Image !== 'undefined') {
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image();
      image.src = url;
      await ensureImageDecoded(image);
      const width = image.naturalWidth || image.width;
      const height = image.naturalHeight || image.height;
      return imageDataFromSource(image, width, height);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  throw new Error('No image decoding APIs are available in this environment.');
};

const toImageData = async (input: WebImageInput): Promise<ImageData> => {
  if (isImageData(input)) return input;
  if (isHtmlCanvas(input) || isOffscreenCanvas(input)) {
    return imageDataFromCanvas(input);
  }
  if (isHtmlImage(input)) {
    await ensureImageDecoded(input);
    const width = input.naturalWidth || input.width;
    const height = input.naturalHeight || input.height;
    return imageDataFromSource(input, width, height);
  }
  if (isImageBitmap(input)) {
    return imageDataFromSource(input, input.width, input.height);
  }
  if (isBlob(input)) {
    return imageDataFromBlob(input);
  }
  throw new Error('Unsupported image input type.');
};

const imageDataToRgb = (data: ImageData, channelOrder: RasterImage['channelOrder']): RasterImage => {
  const { width, height } = data;
  const src = data.data;
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, o = 0; i < src.length; i += 4, o += 3) {
    out[o] = src[i];
    out[o + 1] = src[i + 1];
    out[o + 2] = src[i + 2];
  }
  return {
    data: out,
    width,
    height,
    channels: 3,
    channelOrder,
  };
};

export const loadImage = async (
  input: WebImageInput,
  options: LoadImageOptions = {},
): Promise<RasterImage> => {
  const channelOrder = options.channelOrder ?? 'rgb';
  if (channelOrder !== 'rgb') {
    throw new Error(`Unsupported channel order: ${channelOrder}`);
  }
  const imageData = await toImageData(input);
  return imageDataToRgb(imageData, channelOrder);
};

export const loadGrayscaleImage = async (input: WebImageInput): Promise<RasterImage> => {
  const imageData = await toImageData(input);
  const { width, height } = imageData;
  const src = imageData.data;
  const out = new Uint8Array(width * height);
  for (let i = 0, o = 0; i < src.length; i += 4, o += 1) {
    const r = src[i];
    const g = src[i + 1];
    const b = src[i + 2];
    out[o] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return {
    data: out,
    width,
    height,
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

const toOrtTensor = (ortRuntime: OrtModule, tensor: Tensor): ort.Tensor =>
  new ortRuntime.Tensor(toOrtType(tensor.type), tensor.data, tensor.shape);

const fromOrtTensor = (tensor: ort.Tensor): Tensor => ({
  data: tensor.data as Float32Array | Int32Array | Uint8Array,
  shape: [...tensor.dims],
  type: fromOrtType(tensor.type),
});

class OrtRunner implements InferenceRunner {
  protected session: ort.InferenceSession;
  protected ortRuntime: OrtModule;

  constructor(ortRuntime: OrtModule, session: ort.InferenceSession) {
    this.ortRuntime = ortRuntime;
    this.session = session;
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const ortFeeds: Record<string, ort.Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      ortFeeds[name] = toOrtTensor(this.ortRuntime, tensor);
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

  constructor(ortRuntime: OrtModule, session: ort.InferenceSession, textName: string, linkName: string) {
    super(ortRuntime, session);
    this.textName = textName;
    this.linkName = linkName;
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const ortFeeds: Record<string, ort.Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      ortFeeds[name] = toOrtTensor(this.ortRuntime, tensor);
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

  constructor(ortRuntime: OrtModule, session: ort.InferenceSession, textInputName?: string) {
    super(ortRuntime, session);
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
        ortFeeds[name] = new this.ortRuntime.Tensor('int64', big, tensor.shape);
      } else {
        ortFeeds[name] = toOrtTensor(this.ortRuntime, tensor);
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

const createSession = async (
  ortRuntime: OrtModule,
  model: string | Uint8Array | ArrayBuffer,
  options: LoadSessionOptions = {},
): Promise<ort.InferenceSession> => {
  if (typeof model === 'string') {
    return ortRuntime.InferenceSession.create(model, options.sessionOptions);
  }
  const data = model instanceof Uint8Array ? model : new Uint8Array(model);
  return ortRuntime.InferenceSession.create(data, options.sessionOptions);
};

export const loadSession = async (
  model: string | Uint8Array | ArrayBuffer,
  options: LoadSessionOptions = {},
): Promise<ort.InferenceSession> => {
  const ortRuntime = await loadOrt();
  return createSession(ortRuntime, model, options);
};

export const loadCharset = async (path: string): Promise<string> => {
  if (typeof fetch === 'undefined') {
    throw new Error('fetch is unavailable in this environment.');
  }
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load charset from ${path}: ${response.status} ${response.statusText}`);
  }
  return (await response.text()).trimEnd();
};

export interface DetectorModelOptions extends LoadSessionOptions {
  inputName?: string;
  textOutputName?: string;
  linkOutputName?: string;
}

export const loadDetectorModel = async (
  model: string | Uint8Array | ArrayBuffer,
  options: DetectorModelOptions = {},
): Promise<DetectorModel> => {
  const ortRuntime = await loadOrt();
  const session = await createSession(ortRuntime, model, options);
  const inputName = options.inputName ?? session.inputNames[0];
  const textOutputName = options.textOutputName ?? 'text';
  const linkOutputName = options.linkOutputName ?? 'link';
  if (!inputName || !textOutputName || !linkOutputName) {
    throw new Error('Detector model is missing required input/output names.');
  }
  return {
    session: new CraftRunner(ortRuntime, session, textOutputName, linkOutputName),
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
  model: string | Uint8Array | ArrayBuffer,
  options: RecognizerModelOptions,
): Promise<RecognizerModel> => {
  const ortRuntime = await loadOrt();
  const session = await createSession(ortRuntime, model, options);
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
    session: new RecognizerRunner(ortRuntime, session, textInputName),
    inputName,
    outputName,
    textInputName,
    charset: options.charset,
    symbols: options.symbols,
    blankIndex: options.blankIndex,
  };
};
