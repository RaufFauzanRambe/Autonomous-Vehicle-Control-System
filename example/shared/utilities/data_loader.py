"""CSV / JSON data helpers shared by every example module.

All tabular sample datasets in this project (``*.csv``) follow the same
conventions: a single header row, SI units documented in
``documentation/dataset_description.md``, and one record per row.

Extension point: add parquet/HDF5 readers here once datasets grow beyond
the toy sizes used in the examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a :class:`pandas.DataFrame`.

    Args:
        path: Path to a ``.csv`` file with a header row.

    Returns:
        DataFrame with inferred dtypes.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path)


def load_csv_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load a CSV file as a list of plain dictionaries (one per row).

    Args:
        path: Path to a ``.csv`` file with a header row.

    Returns:
        List of row dictionaries keyed by column name.
    """
    frame = load_csv(path)
    return frame.to_dict(orient="records")


def save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories if needed.

    Args:
        frame: DataFrame to write.
        path: Destination ``.csv`` path.

    Returns:
        The resolved destination path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(file_path, index=False)
    return file_path


def save_json(obj: Any, path: str | Path) -> Path:
    """Serialize ``obj`` to a pretty-printed JSON file.

    NumPy scalar types are converted to built-in Python types first so
    that simulation summaries can be dumped directly.

    Args:
        obj: JSON-serializable object (numpy scalars are converted).
        path: Destination ``.json`` path.

    Returns:
        The resolved destination path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(value: Any) -> Any:
        if hasattr(value, "item"):  # numpy scalar
            return value.item()
        if hasattr(value, "tolist"):  # numpy array
            return value.tolist()
        return str(value)

    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_default)
        handle.write("\n")
    return file_path
