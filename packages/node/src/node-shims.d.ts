declare module 'node:fs/promises' {
  export const readFile: (path: string, encoding?: string) => Promise<string>;
  export const access: (path: string) => Promise<void>;
  export const mkdir: (path: string, options?: { recursive?: boolean }) => Promise<void>;
  export const writeFile: (
    path: string,
    data: string | Uint8Array,
    encoding?: string,
  ) => Promise<void>;
}

declare module 'node:crypto' {
  export interface Hash {
    update: (data: string | Uint8Array) => Hash;
    digest: (encoding: 'hex') => string;
  }
  export const createHash: (algorithm: string) => Hash;
}

declare module 'node:path' {
  export const join: (...parts: string[]) => string;
  export const basename: (p: string, ext?: string) => string;
  export const dirname: (p: string) => string;
  const path: {
    join: (...parts: string[]) => string;
    basename: (p: string, ext?: string) => string;
    dirname: (p: string) => string;
    posix: { join: (...parts: string[]) => string };
  };
  export default path;
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
