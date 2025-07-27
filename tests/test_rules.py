import pandas as pd
from labelqa.rules import apply_rules


def test_numeric_range_rule():
    df = pd.DataFrame({"x": [1, 2, 3, 10]})
    cfg = {
        "rules": [
            {
                "name": "x_between_1_and_3",
                "type": "numeric_range",
                "column": "x",
                "min": 1,
                "max": 3,
                "inclusive": True,
            }
        ]
    }
    results = apply_rules(df, cfg)
    assert len(results) == 1
    r = results[0]
    assert r.status == "fail"
    assert r.n_violations == 1  # the 10


def test_allowed_values_pass():
    df = pd.DataFrame({"label": ["a", "b", "a"]})
    cfg = {
        "rules": [
            {
                "name": "allowed",
                "type": "allowed_values",
                "column": "label",
                "values": ["a", "b"],
            }
        ]
    }
    r = apply_rules(df, cfg)[0]
    assert r.status == "pass"
