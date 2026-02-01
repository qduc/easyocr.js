# easyocr.js

A JavaScript variant of [EasyOCR](https://github.com/JaidedAI/EasyOCR).

## Repository Structure

- `packages/core`: Shared pipeline logic, model abstract interfaces.
- `packages/node`: Node.js runtime using `onnxruntime-node` and `sharp`.
- `packages/web`: Web runtime using `onnxruntime-web` and `Canvas`.
- `packages/cli`: Command line interface.
- `examples`: Usage examples for different environments.
- `benchmarks`: Performance comparisons.

## Development

This is a monorepo using Bun workspaces.

```bash
bun install
bun run build
```
