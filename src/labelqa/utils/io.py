from pathlib import Path
import pandas as pd


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
