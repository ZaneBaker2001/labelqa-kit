from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from labelqa.rules import RuleResult


def _df_samples_for_rule(df: pd.DataFrame, rule: RuleResult, sample_n: int) -> pd.DataFrame:
    details = rule.details
    if rule.type == "duplicate_rows":
        subset = details.get("subset", None)
        dup_mask = df.duplicated(subset=subset, keep=False)
        return df.loc[dup_mask].head(sample_n)

    col = details.get("column")
    if rule.type == "regex":
        # same logic as used in rule computation
        # recompute mask to sample (keeps module stateless)
        # We can't easily recompute pattern here without passing the rule object; skip for simplicity
        # (You can extend RuleResult to carry mask indices if you want rich sampling)
        return pd.DataFrame()
    # For generic ones where the violation mask is not recomputed, just return empty
    return pd.DataFrame()


def generate_reports(
    df: pd.DataFrame,
    rule_results: List[RuleResult],
    schema_results: Dict[str, Any],
    report_html_path: Path | None,
    report_json_path: Path | None,
    sample_violations: int,
    severity_map: Dict[str, str],
):
    serializable = {
        "rules": [
            {
                **asdict(r),
                "severity": severity_map.get(r.type, "INFO"),
            }
            for r in rule_results
        ],
        "schema": schema_results,
        "n_rows": len(df),
    }

    if report_json_path:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    if report_html_path:
        env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            autoescape=select_autoescape(),
        )
        tpl = env.get_template("report.html.j2")

        rendered = tpl.render(
            data=serializable,
            schema_errors=schema_results.get("errors", []),
        )
        report_html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_html_path, "w", encoding="utf-8") as f:
            f.write(rendered)
