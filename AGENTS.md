# Repository Guidelines

## Project Structure & Module Organization

This repo is a small Bun-workspaces monorepo for a JavaScript/TypeScript port of EasyOCR.

- `packages/core/`: shared types and pipeline primitives (published as `@easyocrjs/core`)
- `packages/node/`: Node.js runtime wrapper (published as `@easyocrjs/node`)
- `packages/web/`: browser runtime wrapper (published as `@easyocrjs/web`)
- `packages/cli/`: CLI entrypoint (published as `@easyocrjs/cli`, binary: `easyocr`)
- `examples/`: runnable sample usage
- `benchmarks/`: performance experiments
- `models/`: model assets (do not edit unless you are intentionally updating weights)
- `dist/`: build output (prefer not to edit by hand; regenerate via build)
- `PIPELINE_CONTRACT.md`, `OVERVIEW_PLAN.md`: high-level architecture and pipeline notes

## Build, Test, and Development Commands

From repo root:

- `bun install`: install all workspace dependencies
- `bun run build`: run each workspace build (TypeScript compilation to `dist/`)
- `bun run test`: run tests across workspaces

Target a single workspace when iterating:

- `bun run -F @easyocrjs/node test`
- `bun run -F @easyocrjs/web build`

## Coding Style & Naming Conventions

- Language: TypeScript, ESM (`"type": "module"`).
- Formatting: follow existing code (2-space indent, semicolons, single quotes).
- Naming: packages use `@easyocrjs/*`; tests use `*.test.ts`.
- Avoid committing generated artifacts unless the repo explicitly expects them (prefer source-only changes + rebuild).

## Testing Guidelines

- Framework: Vitest (see `vitest.config.ts`).
- Test locations: `packages/*/test/*.test.ts`.
- Add/adjust tests with behavior changes; there is no enforced coverage threshold in this repo today.

## Commit & Pull Request Guidelines

- There is no established commit-message convention in this repo yet.
- Recommended: use Conventional Commits (`feat: ...`, `fix: ...`, `chore: ...`) and keep messages scoped to one change.
- PRs should include: what/why, how to test (commands run), and any relevant example output or screenshots (especially for `packages/web`).
