import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { fetchModel, loadDetectorModel, loadRecognizerModel } from '../src/index';

const ortMocks = vi.hoisted(() => ({
  createMock: vi.fn(),
}));

vi.mock('onnxruntime-web', () => ({
  InferenceSession: {
    create: ortMocks.createMock,
  },
}));

const originalFetch = globalThis.fetch;

const makeHeaders = (values: Record<string, string>) => ({
  get(name: string) {
    return values[name.toLowerCase()] ?? null;
  },
});

const mockFetchResponse = (response: Record<string, unknown>) => {
  // @ts-expect-error test shim
  globalThis.fetch = vi.fn().mockResolvedValue(response);
};

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.clearAllMocks();
});

beforeEach(() => {
  ortMocks.createMock.mockReset();
});

describe('fetchModel error handling', () => {
  it('throws with status and url for 404 responses', async () => {
    const url = 'https://example.com/models/onnx/english_g2.onnx';
    mockFetchResponse({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      headers: makeHeaders({}),
      async text() {
        return 'Not found';
      },
      async arrayBuffer() {
        return new ArrayBuffer(0);
      },
    });

    await expect(fetchModel(url)).rejects.toThrow(
      `Failed to fetch model from ${url}: 404 Not Found`,
    );
  });

  it('throws with status and url for 500 responses', async () => {
    const url = 'https://example.com/models/onnx/craft_mlt_25k.onnx';
    mockFetchResponse({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      headers: makeHeaders({}),
      async text() {
        return 'Server error';
      },
      async arrayBuffer() {
        return new ArrayBuffer(0);
      },
    });

    await expect(fetchModel(url)).rejects.toThrow(
      `Failed to fetch model from ${url}: 500 Internal Server Error`,
    );
  });

  it('throws on unexpected content types with actionable details', async () => {
    const url = 'https://example.com/models/onnx/english_g2.onnx';
    mockFetchResponse({
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: makeHeaders({
        'content-type': 'text/html',
        'content-length': '1024',
      }),
      async text() {
        return '<html><body>Not a model</body></html>';
      },
      async arrayBuffer() {
        return new ArrayBuffer(0);
      },
    });

    try {
      await fetchModel(url);
      throw new Error('Expected fetchModel to throw.');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      expect(message).toContain(url);
      expect(message).toContain('content-type text/html');
      expect(message).toContain('Expected binary content');
    }
  });

  it('throws on suspiciously small payloads with actionable details', async () => {
    const url = 'https://example.com/models/onnx/english_g2.onnx';
    mockFetchResponse({
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: makeHeaders({
        'content-type': 'application/octet-stream',
        'content-length': '120',
      }),
      async text() {
        return 'Not a model';
      },
      async arrayBuffer() {
        return new ArrayBuffer(0);
      },
    });

    try {
      await fetchModel(url);
      throw new Error('Expected fetchModel to throw.');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      expect(message).toContain(url);
      expect(message).toContain('content-length 120');
      expect(message).toContain('Expected binary content');
    }
  });

  it('wraps arrayBuffer failures with actionable details', async () => {
    const url = 'https://example.com/models/onnx/english_g2.onnx';
    mockFetchResponse({
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: makeHeaders({
        'content-type': 'application/octet-stream',
        'content-length': '5000',
      }),
      async text() {
        return '';
      },
      async arrayBuffer() {
        throw new RangeError('Array buffer allocation failed');
      },
    });

    try {
      await fetchModel(url);
      throw new Error('Expected fetchModel to throw.');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      expect(message).toContain(url);
      expect(message).toContain('Array buffer allocation failed');
    }
  });
});

describe('model load error propagation', () => {
  it('wraps detector model fetch failures with context', async () => {
    const cause = new Error('fetch failed');
    ortMocks.createMock.mockRejectedValueOnce(cause);

    try {
      await loadDetectorModel('https://example.com/models/onnx/craft_mlt_25k.onnx');
      throw new Error('Expected loadDetectorModel to throw.');
    } catch (error) {
      const err = error as Error & { cause?: unknown };
      expect(err.message).toContain('detector model');
      expect(err.message).toContain('fetch failed');
      expect(err.cause).toBe(cause);
    }
  });

  it('wraps recognizer model arrayBuffer failures with context', async () => {
    const cause = new Error('arrayBuffer failed');
    ortMocks.createMock.mockRejectedValueOnce(cause);

    try {
      await loadRecognizerModel('https://example.com/models/onnx/english_g2.onnx', {
        charset: 'abc',
      });
      throw new Error('Expected loadRecognizerModel to throw.');
    } catch (error) {
      const err = error as Error & { cause?: unknown };
      expect(err.message).toContain('recognizer model');
      expect(err.message).toContain('arrayBuffer failed');
      expect(err.cause).toBe(cause);
    }
  });

  it('wraps onnx parse errors with model context', async () => {
    const cause = new Error('ONNX parse error');
    ortMocks.createMock.mockRejectedValueOnce(cause);

    try {
      await loadDetectorModel('https://example.com/models/onnx/craft_mlt_25k.onnx');
      throw new Error('Expected loadDetectorModel to throw.');
    } catch (error) {
      const err = error as Error & { cause?: unknown };
      expect(err.message).toContain('detector model');
      expect(err.message).toContain('ONNX parse error');
      expect(err.cause).toBe(cause);
    }
  });
});
