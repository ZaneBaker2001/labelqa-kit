from pathlib import Path
import json
import typer
from rich import print
from rich.console import Console

from labelqa.utils.io import read_dataframe
from labelqa.config import load_rules_config, load_schema
from labelqa.rules import apply_rules
from labelqa.reporting.report import generate_reports
from labelqa.generators.synthetic_data import generate_synthetic

app = typer.Typer(help="labelqa - Lightweight dataset labeling QA toolkit")
console = Console()


@app.command()
def validate(
    data: Path = typer.Option(..., exists=True, help="CSV/Parquet file to validate"),
    schema: Path = typer.Option(..., exists=True, help="JSON schema file"),
    rules: Path = typer.Option(..., exists=True, help="YAML rules file"),
    report: Path = typer.Option(None, help="Path to write HTML report"),
    report_json: Path = typer.Option(None, help="Path to write JSON report"),
):
    """Validate DATA with SCHEMA and RULES, and optionally emit reports."""
    df = read_dataframe(data)
    schema_obj = load_schema(schema)
    rules_cfg = load_rules_config(rules)

    console.log(f"Loaded dataframe with shape {df.shape}")

    results = apply_rules(df, rules_cfg)

    # Schema validation
    schema_results = schema_obj.validate(df)
    if schema_results["status"] == "fail":
        console.print("[red]Schema validation failed[/red]")
    else:
        console.print("[green]Schema validation passed[/green]")

    if report or report_json:
        generate_reports(
            df=df,
            rule_results=results,
            schema_results=schema_results,
            report_html_path=report,
            report_json_path=report_json,
            sample_violations=rules_cfg.get("global", {}).get("sample_violations", 5),
            severity_map=rules_cfg.get("global", {}).get("severity_map", {}),
        )

    # Exit code: 0 if all pass, 1 otherwise
    any_fail = any(r.status == "fail" for r in results) or schema_results["status"] == "fail"
    if any_fail:
        raise typer.Exit(code=1)
    else:
        console.print("[bold green]All checks passed![/bold green]")


@app.command("generate-synthetic")
def generate_synth(
    schema: Path = typer.Option(..., exists=True, help="JSON schema file"),
    rows: int = typer.Option(1000, help="Number of rows to generate"),
    out: Path = typer.Option(..., help="Where to write the generated CSV"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Generate synthetic data from a schema."""
    schema_obj = load_schema(schema)
    df = generate_synthetic(schema_obj, rows=rows, seed=seed)
    df.to_csv(out, index=False)
    print(f"Wrote synthetic data to {out}")


if __name__ == "__main__":
    app()
