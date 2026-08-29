"""Tidy frames to booktabs LaTeX. The ONLY path by which a number reaches a document."""
from __future__ import annotations

import numpy as np
import pandas as pd

_ESCAPE = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}


def escape(text) -> str:
    """Escape LaTeX specials. CKD feature names contain underscores and would
    otherwise break the build; a raw % would silently comment out a table row."""
    return "".join(_ESCAPE.get(ch, ch) for ch in str(text))


def fmt(value, digits: int = 3) -> str:
    """Every number reaches a document through here, so the missing-value policy
    lives in exactly one place. A literal 'nan' in a submitted table is the most
    embarrassing possible defect."""
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if not np.isfinite(number):
        return "--"
    if number != 0 and abs(number) < 10 ** (-digits):
        return f"{number:.1e}".replace("e-0", "e-").replace("e+0", "e")
    return f"{number:.{digits}f}"


def fmt_ci(point, low, high, digits: int = 3) -> str:
    if point is None or not np.isfinite(float(point) if point is not None else np.nan):
        return "--"
    if low is None or high is None or not (np.isfinite(float(low)) and np.isfinite(float(high))):
        return fmt(point, digits)
    return f"{fmt(point, digits)} [{fmt(low, digits)}, {fmt(high, digits)}]"


def fmt_p(p) -> str:
    """Report a floor rather than 0.000: a p-value is never exactly zero."""
    try:
        value = float(p)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return r"$<$0.001"
    return f"{value:.3f}"


def to_booktabs(df: pd.DataFrame, caption: str, label: str,
                column_format: str | None = None, notes: str | None = None,
                placement: str = "t", small: bool = True,
                escape_cells: bool = False) -> str:
    """A complete table environment.

    Contract: **cells arrive LaTeX-ready.** Table builders call `escape()` on text
    columns and `fmt`/`fmt_ci`/`fmt_p` on numbers, then pass them through untouched.
    That is the only workable split, because a marker type cannot survive a round
    trip through a DataFrame -- pandas coerces a str subclass back to plain str -- so
    "$<$0.001" from fmt_p would be double-escaped by an escaping-by-default table
    writer. Set escape_cells=True only for a frame of raw, unformatted text.
    """
    columns = list(df.columns)
    if column_format is None:
        column_format = "l" + "r" * (len(columns) - 1)
    header = " & ".join(escape(c) for c in columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = [escape(v) if escape_cells else str(v) for v in row.tolist()]
        body_rows.append(" & ".join(cells) + r" \\")
    parts = [
        rf"\begin{{table}}[{placement}]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small" if small else "",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if notes:
        parts.append(rf"\par\vspace{{2pt}}\footnotesize {notes}")
    parts.append(r"\end{table}")
    return "\n".join(part for part in parts if part) + "\n"
