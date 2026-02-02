import { describe, it, expect } from 'vitest';
import { version, loadImage, loadDetectorModel, loadRecognizerModel, loadCharset, fetchModel, getDefaultModelBaseUrl } from '../src/index';

describe('easyocr-js web', () => {
  it('should export version from core', () => {
    expect(version).toBe('0.1.1');
  });

  it('should provide a default, CORS-safe model base URL', () => {
    const base = getDefaultModelBaseUrl();
    expect(base).toContain('media.githubusercontent.com');
    expect(base).toContain(`/v${version}/models`);
  });

  it('fetchModel should throw on Git LFS pointer responses', async () => {
    const originalFetch = globalThis.fetch;
    // @ts-expect-error test shim
    globalThis.fetch = async () => {
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        headers: {
          get(name: string) {
            if (name.toLowerCase() === 'content-type') return 'text/plain';
            if (name.toLowerCase() === 'content-length') return '130';
            return null;
          },
        },
        async text() {
          return [
            'version https://git-lfs.github.com/spec/v1',
            'oid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
            'size 123',
          ].join('\n');
        },
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      } as any;
    };

    await expect(fetchModel('https://raw.githubusercontent.com/qduc/easyocr.js/main/models/onnx/english_g2.onnx'))
      .rejects.toThrow('Git LFS pointer');

    globalThis.fetch = originalFetch;
  });

  it('should export web runtime helpers', () => {
    expect(loadImage).toBeTypeOf('function');
    expect(loadDetectorModel).toBeTypeOf('function');
    expect(loadRecognizerModel).toBeTypeOf('function');
    expect(loadCharset).toBeTypeOf('function');
  });
});
