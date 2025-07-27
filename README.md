# Label QA Kit

**A lightweight, Dockerized dataset labeling QA toolkit** with a pluggable regex/numeric rule engine, schema validation, and synthetic data generation. Built with pandas, NumPy, and Typer.

## Features

- **Rule engine** (YAML): regex, numeric range, allowed values, null-fraction, uniqueness, and more.
- **Schema validation** (JSON Schema via pydantic types).
- **Automated HTML/JSON reporting** (with severity levels, rule summaries, and samples of violations).
- **Synthetic data generator** to bootstrap QA pipelines & demos.
- **Dockerized** for easy sharing with labeling teams.
- **CI-ready** with pytest + pre-commit hooks.

## Quickstart

```bash
# 1) Install
pip install -e .

# 2) Validate a dataset against a schema + rules
labelqa validate \
  --data examples/labels.csv \
  --schema examples/schema.json \
  --rules examples/rules.yml \
  --report out/report.html \
  --report-json out/report.json

# 3) Generate a synthetic dataset from a schema
labelqa generate-synthetic \
  --schema examples/schema.json \
  --rows 1000 \
  --out out/synth.csv

# 4) Run tests
pytest -q

# 5) Docker
docker build -t labelqa-kit .
docker run --rm -v $PWD:/work -w /work labelqa-kit \
  labelqa validate \
    --data examples/labels.csv \
    --schema examples/schema.json \
    --rules examples/rules.yml \
    --report out/report.html
