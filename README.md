# Cy Analysis

Project for analyzing survey data and rewriting report chapters.

# Result
Result markdown file path is
```bash
/outputs/ch45_rewrite/第4章-第5章重写稿_详细版.md
```

## Environment Setup

This project uses `uv` for dependency management.

To sync the environment and create the `.venv/` directory:
```bash
uv sync
```

## Running Scripts

Use `uv run` to execute the analysis and generation scripts:

```bash
# Run data analysis
uv run python models/ch45_rewrite/run_analysis.py

# Generate detailed chapters
uv run python models/ch45_rewrite/generate_detailed_chapters.py
```

## Project Structure

- `data/`: Input data files
- `docs/`: Project documentation and models
- `models/`: Analysis and generation scripts
- `outputs/`: Generated reports and results
- `resources/`: Reference materials (not tracked in git)
- `utils/`: Utility scripts (not tracked in git)

Note: `resources/` and `utils/` directories are intentionally excluded from the repository.
