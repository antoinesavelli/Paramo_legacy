# Copilot Instructions

## Project Guidelines
- Never use print loops that output per-item content (per-file, per-row, per-symbol, etc.). Always aggregate first, then print summary statistics only, with at most 2 example rows of data.
- Test files are located in a 'tests/' folder (not identified by test_*.py / *_test.py filename patterns).
- Use `config/config.py` as the single canonical config file. No new config files should ever be created anywhere in the codebase. All new settings must go into the appropriate dataclass in `config/config.py`. All subsystems should obtain config via `config.loader.build_config()`.