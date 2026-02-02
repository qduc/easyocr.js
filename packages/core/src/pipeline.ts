import type {
  Box,
  DetectorModel,
  OcrOptions,
  OcrResult,
  RasterImage,
  RecognizerModel,
  Tensor,
} from './types.js';
import { buildCrops } from './crop.js';
import { detectorPostprocessDebug, detectorPreprocess, tensorToHeatmap } from './detector.js';
import { ctcGreedyDecode, recognizerPreprocess } from './recognizer.js';
import { mergeOcrResultsByLine } from './postprocess.js';
import { resolveOcrOptions } from './types.js';
import type { TraceWriter } from './trace.js';
import { traceStep } from './trace.js';
import { toFloatImage } from './utils.js';
import { LANGUAGE_CHARS } from './languages.js';

export interface RecognizeInput {
  image: RasterImage;
  recognitionImage?: RasterImage;
  detector: DetectorModel;
  recognizer: RecognizerModel;
  options?: Partial<OcrOptions>;
  trace?: TraceWriter;
}

interface RecognitionCrop {
  box: Box;
  text: string;
  confidence: number;
  rotation: number;
}

const ensureTensor = (tensor?: Tensor): Tensor => {
  if (!tensor) {
    throw new Error('Missing tensor output from model.');
  }
  return tensor;
};

const buildIgnoreIndices = (recognizer: RecognizerModel, options: OcrOptions): number[] => {
  const blankIndex = recognizer.blankIndex ?? 0;
  const ignoreIndices: number[] = [];

  const getOutputIndex = (charsetIndex: number) => {
    if (blankIndex === 0) return charsetIndex + 1;
    return charsetIndex >= blankIndex ? charsetIndex + 1 : charsetIndex;
  };

  if (options.allowlist) {
    const allowSet = new Set(options.allowlist);
    for (let i = 0; i < recognizer.charset.length; i++) {
      if (!allowSet.has(recognizer.charset[i])) {
        ignoreIndices.push(getOutputIndex(i));
      }
    }
    return ignoreIndices;
  }

  if (options.blocklist) {
    const blockSet = new Set(options.blocklist);
    for (let i = 0; i < recognizer.charset.length; i++) {
      if (blockSet.has(recognizer.charset[i])) {
        ignoreIndices.push(getOutputIndex(i));
      }
    }
    return ignoreIndices;
  }

  if (options.langList && options.langList.length > 0) {
    const allowedChars = new Set<string>();
    for (const lang of options.langList) {
      const chars = LANGUAGE_CHARS[lang];
      if (chars) {
        for (const char of chars) {
          allowedChars.add(char);
        }
      }
    }
    // Default symbols from Python EasyOCR
    const symbols = recognizer.symbols ?? '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ';
    for (const char of symbols) {
      allowedChars.add(char);
    }

    for (let i = 0; i < recognizer.charset.length; i++) {
      if (!allowedChars.has(recognizer.charset[i])) {
        ignoreIndices.push(getOutputIndex(i));
      }
    }
    return ignoreIndices;
  }

  return [];
};

export const recognize = async ({
  image,
  recognitionImage,
  detector,
  recognizer,
  options,
  trace,
}: RecognizeInput): Promise<OcrResult[]> => {
  const resolved = resolveOcrOptions(options);
  await traceStep(trace, {
    name: 'ocr_options',
    kind: 'params',
    params: resolved,
  });
  await traceStep(trace, {
    name: 'load_image',
    kind: 'image',
    image,
    meta: {
      width: image.width,
      height: image.height,
      channels: image.channels,
      channelOrder: image.channelOrder,
    },
  });
  const detectorPrep = detectorPreprocess(image, resolved);
  await traceStep(trace, {
    name: 'resize_aspect_ratio',
    kind: 'image',
    image: detectorPrep.resized,
    meta: {
      canvasSize: resolved.canvasSize,
      magRatio: resolved.magRatio,
      resizedWidth: detectorPrep.resized.width,
      resizedHeight: detectorPrep.resized.height,
    },
  });
  await traceStep(trace, {
    name: 'pad_to_stride',
    kind: 'image',
    image: detectorPrep.padded,
    meta: {
      stride: resolved.align,
      padded: detectorPrep.padRight > 0 || detectorPrep.padBottom > 0,
      padRight: detectorPrep.padRight,
      padBottom: detectorPrep.padBottom,
    },
  });
  const floatImage = toFloatImage(detectorPrep.padded, resolved.mean, resolved.std);
  await traceStep(trace, {
    name: 'normalize_mean_variance',
    kind: 'tensor',
    tensor: {
      data: floatImage.data,
      shape: [floatImage.height, floatImage.width, floatImage.channels],
      type: 'float32',
    },
    layout: 'HWC',
    colorSpace: 'RGB',
    meta: {
      mean: resolved.mean,
      std: resolved.std,
      width: floatImage.width,
      height: floatImage.height,
    },
  });
  await traceStep(trace, {
    name: 'to_tensor_layout',
    kind: 'tensor',
    tensor: detectorPrep.input,
    layout: 'NCHW',
    colorSpace: 'RGB',
  });
  await traceStep(trace, {
    name: 'detector_input_final',
    kind: 'tensor',
    tensor: detectorPrep.input,
    layout: 'NCHW',
    colorSpace: 'RGB',
  });
  const detectorOutputs = await detector.session.run({ [detector.inputName]: detectorPrep.input });
  const textTensor = ensureTensor(detectorOutputs[detector.textOutputName]);
  const linkTensor = ensureTensor(detectorOutputs[detector.linkOutputName]);
  const textMap = tensorToHeatmap(textTensor);
  const linkMap = tensorToHeatmap(linkTensor);
  await traceStep(trace, {
    name: 'detector_raw_output_text',
    kind: 'tensor',
    tensor: { data: textMap.data, shape: [textMap.height, textMap.width], type: 'float32' },
    layout: 'HW',
  });
  await traceStep(trace, {
    name: 'detector_raw_output_link',
    kind: 'tensor',
    tensor: { data: linkMap.data, shape: [linkMap.height, linkMap.width], type: 'float32' },
    layout: 'HW',
  });
  await traceStep(trace, {
    name: 'heatmap_text',
    kind: 'tensor',
    tensor: { data: textMap.data, shape: [textMap.height, textMap.width], type: 'float32' },
    layout: 'HW',
  });
  await traceStep(trace, {
    name: 'heatmap_link',
    kind: 'tensor',
    tensor: { data: linkMap.data, shape: [linkMap.height, linkMap.width], type: 'float32' },
    layout: 'HW',
  });
  // Match Python EasyOCR: boxes are in scoremap space (ratio_net=2), scale back using the resize ratio only.
  // scaleX/Y here are "heatmap -> original" divisors (i.e., x / scaleX = x_original).
  const scaleX = detectorPrep.scaleX / 2;
  const scaleY = detectorPrep.scaleY / 2;
  const { rawBoxesHeatmap, rawBoxesAdjusted, horizontalList, freeList } = detectorPostprocessDebug(
    textMap,
    linkMap,
    resolved,
    scaleX,
    scaleY,
  );
  await traceStep(trace, {
    name: 'threshold_and_box_decode',
    kind: 'boxes',
    boxes: rawBoxesHeatmap,
    meta: {
      coordSpace: 'heatmap',
      scaleX,
      scaleY,
    },
  });
  await traceStep(trace, {
    name: 'adjust_coordinates_to_original',
    kind: 'boxes',
    boxes: rawBoxesAdjusted,
    meta: {
      coordSpace: 'image',
      scaleX,
      scaleY,
    },
  });
  await traceStep(trace, { name: 'detector_boxes_horizontal', kind: 'boxes', boxes: horizontalList });
  await traceStep(trace, { name: 'detector_boxes_free', kind: 'boxes', boxes: freeList });
  // Match Python EasyOCR CPU ordering: process horizontal boxes first, then free boxes.
  const ordered = [...horizontalList, ...freeList];
  await traceStep(trace, { name: 'detector_boxes_ordered', kind: 'boxes', boxes: ordered });
  const recognitionSource = recognitionImage ?? image;
  const crops = buildCrops(recognitionSource, horizontalList, freeList, resolved);
  const results: RecognitionCrop[] = [];
  const ignoreIndices = buildIgnoreIndices(recognizer, resolved);

  for (const crop of crops) {
    const prep = recognizerPreprocess(crop.image, resolved);
    const feeds: Record<string, Tensor> = { [recognizer.inputName]: prep.input };
    if (recognizer.textInputName) {
      feeds[recognizer.textInputName] = {
        data: new Int32Array([0]),
        shape: [1, 1],
        type: 'int32',
      };
    }
    const outputs = await recognizer.session.run(feeds);
    const logitsTensor = ensureTensor(outputs[recognizer.outputName]);
    if (logitsTensor.type !== 'float32') {
      throw new Error('Recognizer output must be float32.');
    }
    const shape = logitsTensor.shape;
    const steps = shape[shape.length - 2];
    const classes = shape[shape.length - 1];
    const decoded = ctcGreedyDecode(
      logitsTensor.data as Float32Array,
      steps,
      classes,
      recognizer.charset,
      recognizer.blankIndex ?? 0,
      ignoreIndices,
    );
    results.push({
      box: crop.box,
      text: decoded.text,
      confidence: decoded.confidence,
      rotation: crop.rotation,
    });
  }

  const rawResults = results.map((result) => ({
    box: result.box,
    text: result.text,
    confidence: result.confidence,
  }));
  await traceStep(trace, {
    name: 'recognizer_results_pre_merge',
    kind: 'boxes',
    boxes: rawResults.map((r) => r.box),
  });

  if (!resolved.mergeLines) {
    return rawResults;
  }

  const rotationOrder: number[] = [];
  const byRotation = new Map<number, OcrResult[]>();
  for (const result of results) {
    if (!byRotation.has(result.rotation)) {
      byRotation.set(result.rotation, []);
      rotationOrder.push(result.rotation);
    }
    byRotation.get(result.rotation)?.push({
      box: result.box,
      text: result.text,
      confidence: result.confidence,
    });
  }

  const merged = rotationOrder.flatMap((rotation) => {
    const group = byRotation.get(rotation) ?? [];
    return mergeOcrResultsByLine(group, resolved);
  });

  await traceStep(trace, {
    name: 'recognizer_results_post_merge',
    kind: 'boxes',
    boxes: merged.map((r) => r.box),
  });

  return merged;
};
