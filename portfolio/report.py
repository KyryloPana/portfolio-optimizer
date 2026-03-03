from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class OutputDirs:
    """
    Canonical output directory structure for reproducible artifacts.
    """
    base: Path
    figures: Path
    tables: Path
    reports: Path
    logs: Path


def ensure_output_dirs(base_dir: str = "output") -> dict[str, Path]:
    """
    Create (if needed) and return standard output directories.

    Structure:
      output/
        figures/
        tables/
        reports/
        logs/
    """
    base = Path(base_dir)
    figs = base / "figures"
    tables = base / "tables"
    reports = base / "reports"
    logs = base / "logs"

    for d in (figs, tables, reports, logs):
        d.mkdir(parents=True, exist_ok=True)

    return {"base": base, "figures": figs, "tables": tables, "reports": reports, "logs": logs}


def ensure_output_dirs_typed(base_dir: str = "output") -> OutputDirs:
    """
    Typed alternative to ensure_output_dirs(). Use if you prefer IDE/type safety.
    """
    d = ensure_output_dirs(base_dir)
    return OutputDirs(
        base=d["base"],
        figures=d["figures"],
        tables=d["tables"],
        reports=d["reports"],
        logs=d["logs"],
    )


def timestamp_tag() -> str:
    """
    Timestamp tag used in filenames for uniqueness + traceability.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_figure(fig: Any, path: Path, dpi: int = 160) -> None:
    """
    Save a Matplotlib figure with consistent formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        # some figures may already be tight or not support it
        pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def save_table(df: pd.DataFrame, path: Path, index: bool = True, float_format: Optional[str] = None) -> None:
    """
    Save a DataFrame to CSV. Defaults to including index for traceability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, float_format=float_format)


def write_text_report(lines: list[str], path: Path) -> None:
    """
    Write a plain-text report, one line per entry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")