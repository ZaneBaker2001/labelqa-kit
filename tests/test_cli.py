import json
from pathlib import Path
import pandas as pd
from typer.testing import CliRunner

from labelqa.cli import app


def test_validate_command(tmp_path: Path):
    runner = CliRunner()

    # Create tiny dataset, schema, rules
    df = pd.DataFrame({"id": [1, 2], "text": ["ok", "bad"], "label": ["positive", "neutral"]})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        '{"columns": {"id": "int64", "text": "string", "label": "string"}, "required": ["id", "text", "label"]}'
    )

    rules_path = tmp_path / "rules.yml"
    rules_path.write_text(
        """
        rules:
          - name: allowed
            type: allowed_values
            column: label
            values: [positive, neutral]
        """
    )

    json_report = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "validate",
            "--data",
            str(data_path),
            "--schema",
            str(schema_path),
            "--rules",
            str(rules_path),
            "--report-json",
            str(json_report),
        ],
    )

    assert result.exit_code == 0
    obj = json.loads(json_report.read_text())
    assert "rules" in obj
