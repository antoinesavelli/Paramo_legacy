# Copilot Instructions

## Project Guidelines
- Never use print loops that output per-item content (per-file, per-row, per-symbol, etc.). Always aggregate first, then print summary statistics only, with at most 2 example rows of data.
- Test files are located in a 'tests/' folder (not identified by test_*.py / *_test.py filename patterns).
- Use `config/config.py` as the single canonical config file. No new config files should ever be created anywhere in the codebase. All new settings must go into the appropriate dataclass in `config/config.py`. All subsystems should obtain config via `config.loader.build_config()`.

## File Handling Guidelines
- Never make recommendations or draw conclusions about the codebase based on failed or empty tool results. If a file cannot be read, explicitly state so and ask the user to provide the content. Do not infer that a file is empty, missing, or problematic just because a tool returned no output. Always verify with a successful read before making any claim about a file or folder.