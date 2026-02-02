import { describe, expect, it } from 'vitest';
import { LANGUAGE_CHARS, guessModel } from '../src/languages';

describe('languages', () => {
  it('has character lists for common languages', () => {
    expect(LANGUAGE_CHARS['en']).toBeDefined();
    expect(LANGUAGE_CHARS['en']).toContain('A');
    expect(LANGUAGE_CHARS['en']).toContain('z');

    expect(LANGUAGE_CHARS['ch_sim']).toBeDefined();
    expect(LANGUAGE_CHARS['ja']).toBeDefined();
    expect(LANGUAGE_CHARS['ko']).toBeDefined();
  });

  it('guesses the correct model for given languages', () => {
    expect(guessModel(['en'])).toBe('english_g2');
    expect(guessModel(['ja'])).toBe('japanese_g2');
    expect(guessModel(['ko', 'en'])).toBe('korean_g2');
    expect(guessModel(['ch_sim'])).toBe('zh_sim_g2');
    expect(guessModel(['ru'])).toBe('cyrillic_g2');
    expect(guessModel(['ar'])).toBe('arabic_g1');
    expect(guessModel(['hi'])).toBe('devanagari_g1');
    expect(guessModel(['bn'])).toBe('bengali_g1');
    expect(guessModel(['fr', 'de'])).toBe('latin_g2');
  });
});
