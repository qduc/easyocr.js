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
import { detectorPostprocess, detectorPreprocess, tensorToHeatmap, orderBoxes } from './detector.js';
import { ctcGreedyDecode, recognizerPreprocess } from './recognizer.js';
import { resolveOcrOptions } from './types.js';

export interface RecognizeInput {
  image: RasterImage;
  detector: DetectorModel;
  recognizer: RecognizerModel;
  options?: Partial<OcrOptions>;
}

interface RecognitionCrop {
  box: Box;
  text: string;
  confidence: number;
}

const ensureTensor = (tensor?: Tensor): Tensor => {
  if (!tensor) {
    throw new Error('Missing tensor output from model.');
  }
  return tensor;
};

export const recognize = async ({
  image,
  detector,
  recognizer,
  options,
}: RecognizeInput): Promise<OcrResult[]> => {
  const resolved = resolveOcrOptions(options);
  const detectorPrep = detectorPreprocess(image, resolved);
  const detectorOutputs = await detector.session.run({ [detector.inputName]: detectorPrep.input });
  const textTensor = ensureTensor(detectorOutputs[detector.textOutputName]);
  const linkTensor = ensureTensor(detectorOutputs[detector.linkOutputName]);
  const textMap = tensorToHeatmap(textTensor);
  const linkMap = tensorToHeatmap(linkTensor);
  const { horizontalList, freeList } = detectorPostprocess(
    textMap,
    linkMap,
    resolved,
    detectorPrep.scaleX,
    detectorPrep.scaleY,
  );
  const ordered = orderBoxes([...horizontalList, ...freeList], resolved);
  const crops = buildCrops(image, ordered, [], resolved);
  const results: RecognitionCrop[] = [];

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
    );
    results.push({ box: crop.box, text: decoded.text, confidence: decoded.confidence });
  }

  return results.map((result) => ({
    box: result.box,
    text: result.text,
    confidence: result.confidence,
  }));
};
