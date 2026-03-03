from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd


def ensure_output_dirs(base_dir: str = "output") -> dict[str, Path]:
    base = Path(base_dir)
    figs = base / "figures"
    tables = base / "tables"

    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    return {"base": base, "figures": figs, "tables": tables}


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_figure(fig, path: Path, dpi: int = 150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def save_table(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=True)


def write_text_report(lines: list[str], path: Path):
    path.write_text("\n".join(lines), encoding="utf-8")
