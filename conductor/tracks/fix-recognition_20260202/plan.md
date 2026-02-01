# Plan - Fix Recognition Discrepancy (Screenshot_20260201_193653)

## Phase 1: Reproduction and Diagnosis
- [x] Task: Create a failing integration test that compares Node.js output against the Python reference for `Screenshot_20260201_193653.png` 5eae576
    - [x] Add a new test file in `packages/node/test/repro_issue.test.ts`
    - [x] Load `Screenshot_20260201_193653.png` and compare result with `python_reference/validation/expected/Screenshot_20260201_193653.json`
    - [x] Confirm the test fails (Red Phase)
- [x] Task: Diagnose the root cause of the recognition discrepancy
    - [x] Compare image crops passed to the recognizer between Python and Node.js
    - [x] Verify normalization and resizing logic in `packages/core/src/recognizer.ts`
    - [x] Inspect the character decoding (greedy decoder) logic
- [x] Task: Conductor - User Manual Verification 'Phase 1: Reproduction and Diagnosis' (Protocol in workflow.md)

## Phase 2: Implementation of Fix
- [x] Task: Implement fix to align Node.js recognition with Python reference
    - [x] Apply necessary changes to `packages/core/src/recognizer.ts` or `packages/core/src/utils.ts`
    - [x] Verify that the reproduction test now passes (Green Phase)
- [x] Task: Refactor and Verify c3c8a1a
    - [x] Clean up any debug logging or temporary diagnostic code
    - [x] Ensure all existing tests in `@easyocrjs/core` and `@easyocrjs/node` pass
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation of Fix' (Protocol in workflow.md)
