import type { Box, RasterImage, Tensor } from './types.js';

export type TraceKind = 'image' | 'tensor' | 'boxes' | 'params';

export interface TraceStepBase {
  name: string;
  kind: TraceKind;
  meta?: Record<string, unknown>;
}

export interface TraceStepImage extends TraceStepBase {
  kind: 'image';
  image: RasterImage;
}

export interface TraceStepTensor extends TraceStepBase {
  kind: 'tensor';
  tensor: Tensor;
  layout?: 'HWC' | 'CHW' | 'NHWC' | 'NCHW' | string;
  colorSpace?: 'RGB' | 'BGR' | 'GRAY' | 'RGBA' | string;
}

export interface TraceStepBoxes extends TraceStepBase {
  kind: 'boxes';
  boxes: Box[];
}

export interface TraceStepParams extends TraceStepBase {
  kind: 'params';
  params: unknown;
}

export type TraceStepInput = TraceStepImage | TraceStepTensor | TraceStepBoxes | TraceStepParams;

export interface TraceWriter {
  addStep(step: TraceStepInput): void | Promise<void>;
}

export const traceStep = async (writer: TraceWriter | undefined, step: TraceStepInput): Promise<void> => {
  if (!writer) return;
  await writer.addStep(step);
};

