# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Remove CLI subprocess path, SDK-only for Claude Code provider

## Context

Benchmark results confirmed SDK path is 42-76% faster across all single-request workloads (warm call: -75.6%, cold start: -66.8%, multi-turn: -64.5%). The `use_sdk` toggle and CLI subprocess inference path are now dead weight. Making `claude-agent-sdk` a hard dependency per user decision.

Auth checking (`_check_auth_fast`, `_check_auth_slow`) still uses lightweight CLI subprocess ca...

### Prompt 2

Great. we can remove benchmarking script. Have we completely switched CLI to SDK for Claude?

### Prompt 3

Great. Update the README.md file.

### Prompt 4

Great. Let us commit all the changes and lets publish a new package.

### Prompt 5

[Request interrupted by user]

### Prompt 6

Great. Let us commit all the changes and lets publish a new version of the package.

