import { describe, expect, it } from 'vitest';
import { getSupportedLanguages, resolveModelForLanguage } from '../src/models';

describe('models helpers', () => {
  it('lists supported languages with models', () => {
    const langs = getSupportedLanguages();
    expect(langs.length).toBeGreaterThan(10);
    const en = langs.find((l) => l.code === 'en');
    expect(en).toBeDefined();
    expect(en?.model).toBe('english_g2');
  });

  it('resolves model + charset for a language', () => {
    const resolved = resolveModelForLanguage('en');
    expect(resolved).toEqual({
      model: 'english_g2',
      charset: 'english_g2.charset.txt',
    });
    expect(resolved).not.toHaveProperty('textInputName');
    expect(resolveModelForLanguage('zh-cn').model).toBe('zh_sim_g2');
    expect(resolveModelForLanguage('jp').model).toBe('japanese_g2');
  });

  it('throws on unsupported language', () => {
    expect(() => resolveModelForLanguage('xx')).toThrow('Unsupported language');
  });
});
