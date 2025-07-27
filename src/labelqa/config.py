from __future__ import annotations
from pathlib import Path
import json
import yaml
import pandas as pd
from typing import Dict, Any
from pydantic import BaseModel, Field


class Schema(BaseModel):
    columns: Dict[str, str]
    required: list[str] = Field(default_factory=list)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        errors = []
        # Column presence
        for col in self.required:
            if col not in df.columns:
                errors.append({"type": "missing_required_column", "column": col})

        # Type checks (best-effort via pandas dtypes name)
        for col, expected_dtype in self.columns.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                # pandas uses object for strings often
                if expected_dtype == "string" and actual not in ("string", "object"):
                    errors.append(
                        {
                            "type": "wrong_dtype",
                            "column": col,
                            "expected": expected_dtype,
                            "actual": actual,
                        }
                    )
                elif expected_dtype != "string" and not actual.startswith(expected_dtype):
                    # loose comparison (e.g., int64 vs int32) – you can tighten if needed
                    if expected_dtype not in actual:
                        errors.append(
                            {
                                "type": "wrong_dtype",
                                "column": col,
                                "expected": expected_dtype,
                                "actual": actual,
                            }
                        )

        return {"status": "fail" if errors else "pass", "errors": errors}


def load_schema(path: Path) -> Schema:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return Schema(**obj)


def load_rules_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
