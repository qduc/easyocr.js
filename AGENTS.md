# Repository Guidelines

## Project Structure & Module Organization

This repo is a small Bun-workspaces monorepo for a JavaScript/TypeScript port of EasyOCR.

- `packages/core/`: shared types and pipeline primitives (published as `@easyocrjs/core`)
- `packages/node/`: Node.js runtime wrapper (published as `@easyocrjs/node`)
- `packages/web/`: browser runtime wrapper (published as `@easyocrjs/web`; currently mostly a re-export of core)
- `packages/cli/`: CLI entrypoint (published as `@easyocrjs/cli`, binary: `easyocr`)
- `examples/`: runnable sample usage
- `benchmarks/`: performance experiments (optional; may not exist in every checkout)
- `models/`: model assets + export scripts (do not edit weights unless you are intentionally updating them)
- `python_reference/`: Python reference + utilities for exporting/validating model behavior
- `python_reference/EasyOCR/`: original EasyOCR code for reference (copied from EasyOCR repo; do not edit here)
- `packages/*/dist/`: per-package build output (generated; ignored by git)
- `PIPELINE_CONTRACT.md`, `OVERVIEW_PLAN.md`: high-level architecture and pipeline notes

## Build, Test, and Development Commands

From repo root:

- `bun install`: install all workspace dependencies
- `bun run build`: run each workspace build (TypeScript compilation to `dist/`)
- `bun run test`: run tests across workspaces

Target a single workspace when iterating:

- `bun run -F @easyocrjs/node test`
- `bun run -F @easyocrjs/web build`

For python environment, we use uv, create an venv in project root:

- `uv venv`: create virtual environment
- `uv pip install`: install python dependencies
- `uv run <script>`: run python script in venv

## Coding Style & Naming Conventions

- Language: TypeScript, ESM (`"type": "module"`).
- Formatting: follow existing code (2-space indent, semicolons, single quotes).
- Naming: packages use `@easyocrjs/*`; tests use `*.test.ts`.
- Avoid committing generated artifacts unless the repo explicitly expects them (prefer source-only changes + rebuild), especially `packages/*/dist/` and `models/onnx/`.

## Testing Guidelines

- Framework: Vitest (see `vitest.config.ts`).
- Test locations: `packages/*/test/*.test.ts`.
- Note: `packages/cli` currently has no real tests (its `test` script is a placeholder).
- Add/adjust tests with behavior changes; there is no enforced coverage threshold in this repo today.

## Debugging Accuracy Drifts Against Python Reference

See `python_reference/validation/README.md` for detailed instructions on generating and diffing traces between the Python EasyOCR and this JS port.

## Model Export (Python)

- PyTorch weights live under `models/` and ONNX exports go to `models/onnx/` (generated).
- Export + validate with `python models/export_onnx.py --detector --recognizer --validate` (see `models/README.md`).

## Commit & Pull Request Guidelines

- There is no established commit-message convention in this repo yet.
- Recommended: use Conventional Commits (`feat: ...`, `fix: ...`, `chore: ...`) and keep messages scoped to one change.
- PRs should include: what/why, how to test (commands run), and any relevant example output or screenshots (especially for `packages/web`).
