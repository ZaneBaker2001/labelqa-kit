from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Callable
import pandas as pd
import numpy as np


@dataclass
class RuleResult:
    name: str
    type: str
    status: str  # pass / fail
    n_violations: int
    fraction_violations: float
    details: Dict[str, Any]


# ---------------------------
# Rule implementations
# ---------------------------

def rule_regex(df: pd.DataFrame, rule: Dict[str, Any]) -> RuleResult:
    col = rule["column"]
    pattern = rule["pattern"]
    flags = rule.get("flags", "")
    fail_on_match = rule.get("fail_on_match", False)

    re_flags = 0
    if "IGNORECASE" in flags:
        re_flags |= re.IGNORECASE

    series = df[col].astype(str).fillna("")
    matches = series.str.match(pattern, flags=re_flags)
    if fail_on_match:
        mask = matches  # matching rows are violations
    else:
        mask = ~matches  # non-matching rows are violations

    n = int(mask.sum())
    frac = float(n) / len(df) if len(df) else 0.0
    return RuleResult(
        name=rule["name"],
        type=rule["type"],
        status="fail" if n > 0 else "pass",
        n_violations=n,
        fraction_violations=frac,
        details={"column": col},
    )


def rule_numeric_range(df: pd.DataFrame, rule: Dict[str, Any]) -> RuleResult:
    col = rule["column"]
    min_v = rule.get("min", -np.inf)
    max_v = rule.get("max", np.inf)
    inclusive = rule.get("inclusive", True)

    series = pd.to_numeric(df[col], errors="coerce")
    if inclusive:
        mask = (series < min_v) | (series > max_v)
    else:
        mask = (series <= min_v) | (series >= max_v)

    n = int(mask.sum())
    frac = float(n) / len(df) if len(df) else 0.0
    return RuleResult(
        name=rule["name"],
        type=rule["type"],
        status="fail" if n > 0 else "pass",
        n_violations=n,
        fraction_violations=frac,
        details={"column": col, "min": min_v, "max": max_v, "inclusive": inclusive},
    )


def rule_allowed_values(df: pd.DataFrame, rule: Dict[str, Any]) -> RuleResult:
    col = rule["column"]
    values = set(rule["values"])
    mask = ~df[col].isin(values)
    n = int(mask.sum())
    frac = float(n) / len(df) if len(df) else 0.0
    return RuleResult(
        name=rule["name"],
        type=rule["type"],
        status="fail" if n > 0 else "pass",
        n_violations=n,
        fraction_violations=frac,
        details={"column": col, "values": list(values)},
    )


def rule_null_fraction(df: pd.DataFrame, rule: Dict[str, Any]) -> RuleResult:
    col = rule["column"]
    max_fraction = rule["max_fraction"]
    frac_null = df[col].isna().mean()
    status = "fail" if frac_null > max_fraction else "pass"
    return RuleResult(
        name=rule["name"],
        type=rule["type"],
        status=status,
        n_violations=int(frac_null * len(df)),
        fraction_violations=float(frac_null),
        details={"column": col, "max_fraction": max_fraction},
    )


def rule_unique_fraction(df: pd.DataFrame, rule: Dict[str, Any]) -> RuleResult:
    col = rule["column"]
    min_fraction = rule["min_fraction"]
    uniq_frac = df[col].nunique(dropna=False) / len(df) if len(df) else 0.0
    status = "fail" if uniq_frac < min_fraction else "pass"
    n_viol = len(df) - df[col].nunique(dropna=False)
    return RuleResult(
        name=rule["name"],
        type=rule["type"],
        status=status,
        n_violations=int(n_viol),
        fraction_violations=1 - float(uniq_frac),
        details={"column": col, "min_fraction": min_fraction},
    )


def rule_length_range(df: pd.DataFrame, rule: Dict[str, Any]) -> RuleResult:
    col = rule["column"]
    min_len = rule.get("min_len", 0)
    max_len = rule.get("max_len", float("inf"))
    lengths = df[col].astype(str).str.len()
    mask = (lengths < min_len) | (lengths > max_len)
    n = int(mask.sum())
    frac = float(n) / len(df) if len(df) else 0.0
    return RuleResult(
        name=rule["name"],
        type=rule["type"],
        status="fail" if n > 0 else "pass",
        n_violations=n,
        fraction_violations=frac,
        details={"column": col, "min_len": min_len, "max_len": max_len},
    )


def rule_duplicate_rows(df: pd.DataFrame, rule: Dict[str, Any]) -> RuleResult:
    subset = rule.get("subset", None)
    max_fraction = rule.get("max_fraction", 0.0)
    dup_mask = df.duplicated(subset=subset, keep=False)
    frac = dup_mask.mean()
    status = "fail" if frac > max_fraction else "pass"
    return RuleResult(
        name=rule["name"],
        type=rule["type"],
        status=status,
        n_violations=int(dup_mask.sum()),
        fraction_violations=float(frac),
        details={"subset": subset, "max_fraction": max_fraction},
    )


RULE_DISPATCH: Dict[str, Callable[[pd.DataFrame, Dict[str, Any]], RuleResult]] = {
    "regex": rule_regex,
    "numeric_range": rule_numeric_range,
    "allowed_values": rule_allowed_values,
    "null_fraction": rule_null_fraction,
    "unique_fraction": rule_unique_fraction,
    "length_range": rule_length_range,
    "duplicate_rows": rule_duplicate_rows,
}


def apply_rules(df: pd.DataFrame, rules_cfg: Dict[str, Any]) -> List[RuleResult]:
    results: List[RuleResult] = []
    for rule in rules_cfg.get("rules", []):
        rtype = rule["type"]
        if rtype not in RULE_DISPATCH:
            raise ValueError(f"Unknown rule type: {rtype}")
        result = RULE_DISPATCH[rtype](df, rule)
        results.append(result)
    return results
