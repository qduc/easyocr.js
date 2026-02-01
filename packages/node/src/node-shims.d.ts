declare module 'node:fs/promises' {
  export const readFile: (path: string, encoding?: string) => Promise<string>;
  export const access: (path: string) => Promise<void>;
}

declare module 'node:url' {
  export const fileURLToPath: (url: string | URL) => string;
}

declare class Buffer extends Uint8Array {
  static from(data: Uint8Array | ArrayBufferLike): Buffer;
}

declare const process: {
  cwd: () => string;
  argv: string[];
  exit: (code?: number) => void;
};
