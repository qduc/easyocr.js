export type ChannelOrder = 'rgb' | 'rgba' | 'bgr' | 'bgra' | 'gray';

export interface RasterImage {
  data: Uint8Array;
  width: number;
  height: number;
  channels: 1 | 3 | 4;
  channelOrder: ChannelOrder;
}

export type Point = [number, number];
export type Box = [Point, Point, Point, Point];

export interface OcrResult {
  box: Box;
  text: string;
  confidence: number;
}

export type TensorType = 'float32' | 'int32' | 'uint8';

export interface Tensor {
  data: Float32Array | Int32Array | Uint8Array;
  shape: number[];
  type: TensorType;
}

export interface InferenceRunner {
  run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
}

export interface DetectorModel {
  session: InferenceRunner;
  inputName: string;
  textOutputName: string;
  linkOutputName: string;
}

export interface RecognizerModel {
  session: InferenceRunner;
  inputName: string;
  outputName: string;
  textInputName?: string;
  charset: string;
  symbols?: string;
  blankIndex?: number;
}

export interface OcrOptions {
  canvasSize: number;
  magRatio: number;
  align: number;
  mean: [number, number, number];
  std: [number, number, number];
  textThreshold: number;
  lowText: number;
  linkThreshold: number;
  minSize: number;
  slopeThs: number;
  ycenterThs: number;
  heightThs: number;
  widthThs: number;
  addMargin: number;
  paragraph: boolean;
  xThs: number;
  yThs: number;
  mergeLines: boolean;
  maxAngleDeg: number;
  rotationInfo: number[];
  contrastThs: number;
  adjustContrast: number;
  langList?: string[];
  allowlist?: string;
  blocklist?: string;
  decoder: 'greedy';
  recognizer: {
    inputHeight: number;
    inputWidth: number;
    inputChannels: 1 | 3;
    mean: number;
    std: number;
  };
}

export const DEFAULT_OCR_OPTIONS: OcrOptions = {
  canvasSize: 2560,
  magRatio: 1.0,
  align: 32,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  textThreshold: 0.7,
  lowText: 0.4,
  linkThreshold: 0.4,
  minSize: 20,
  slopeThs: 0.1,
  ycenterThs: 0.5,
  heightThs: 0.5,
  widthThs: 0.5,
  addMargin: 0.1,
  paragraph: false,
  xThs: 1.0,
  yThs: 0.5,
  mergeLines: true,
  maxAngleDeg: 10,
  rotationInfo: [],
  contrastThs: 0.1,
  adjustContrast: 0.5,
  langList: undefined,
  decoder: 'greedy',
  recognizer: {
    inputHeight: 64,
    inputWidth: 100,
    inputChannels: 1,
    mean: 0.5,
    std: 0.5,
  },
};

export const resolveOcrOptions = (options?: Partial<OcrOptions>): OcrOptions => {
  if (!options) {
    return DEFAULT_OCR_OPTIONS;
  }
  return {
    ...DEFAULT_OCR_OPTIONS,
    ...options,
    recognizer: {
      ...DEFAULT_OCR_OPTIONS.recognizer,
      ...(options.recognizer ?? {}),
    },
  };
};
