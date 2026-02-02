import { ALL_LANGS, guessModel } from './languages.js';

export interface SupportedLanguage {
  code: string;
  label?: string;
  model: string;
}

export interface ResolvedModel {
  model: string;
  charset: string;
  textInputName: string;
}

const LANGUAGE_ALIASES: Record<string, string> = {
  // English
  eng: 'en',

  // Chinese
  zh: 'ch_sim',
  'zh_cn': 'ch_sim',
  'zh_hans': 'ch_sim',
  'zh_sg': 'ch_sim',

  'zh_tw': 'ch_tra',
  'zh_hant': 'ch_tra',
  'zh_hk': 'ch_tra',

  // Japanese / Korean
  jp: 'ja',
  jpn: 'ja',
  kr: 'ko',
  kor: 'ko',

  // Russian
  rus: 'ru',
};

const normalizeLangCode = (lang: string): string => {
  const trimmed = lang.trim();
  if (!trimmed) return '';
  // normalize common separators + casing
  const normalized = trimmed.toLowerCase().replace(/[-\s]+/g, '_');
  return LANGUAGE_ALIASES[normalized] ?? normalized;
};

export const resolveModelForLanguage = (lang: string): ResolvedModel => {
  const code = normalizeLangCode(lang);
  if (!code) {
    throw new Error('resolveModelForLanguage requires a non-empty lang code.');
  }

  if (!ALL_LANGS.includes(code)) {
    throw new Error(
      `Unsupported language "${lang}" (normalized: "${code}"). ` +
        `Use getSupportedLanguages() to list valid codes.`,
    );
  }

  const model = guessModel([code]);
  return {
    model,
    charset: `${model}.charset.txt`,
    textInputName: 'text',
  };
};

export const getSupportedLanguages = (): SupportedLanguage[] => {
  const unique = Array.from(new Set(ALL_LANGS));
  unique.sort((a, b) => a.localeCompare(b));
  return unique.map((code) => ({
    code,
    model: guessModel([code]),
  }));
};
