from __future__ import annotations
import random
import numpy as np
import pandas as pd
from typing import Any

from labelqa.config import Schema


# Simple heuristics for fake data generation
STRING_SAMPLES = [
    "Great product!", "Awful experience", "meh", "Absolutely loved it!",
    "This was fine", "Could be better", "I want my money back",
]
LABEL_SAMPLES = ["positive", "neutral", "negative"]


def generate_synthetic(schema: Schema, rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    cols = list(schema.columns.keys())
    data: dict[str, Any] = {}
    for col in cols:
        dtype = schema.columns[col]
        if dtype.startswith("int"):
            data[col] = np.random.randint(0, 1_000_000, size=rows)
        elif dtype.startswith("float"):
            data[col] = np.random.randn(rows)
        elif dtype == "string":
            if col == "label":
                data[col] = np.random.choice(LABEL_SAMPLES, size=rows)
            elif col == "text":
                data[col] = np.random.choice(STRING_SAMPLES, size=rows)
            else:
                data[col] = [f"str_{i}" for i in range(rows)]
        else:
            # default to object/strings
            data[col] = [f"val_{i}" for i in range(rows)]

    df = pd.DataFrame(data)
    return df
