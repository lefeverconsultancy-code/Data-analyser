"""Read CSV, Excel and pasted tables into a Dataset.

All cells are first read as text and then parsed column by column, so Dutch and English
formats are handled the same way: decimal comma or point, thousands separators, currency
signs, day-first dates. Excel is an import format only.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.dataset import CleaningLog, Dataset
from .clean import is_missing, parse_frame

SEPARATORS = (";", ",", "\t", "|")
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")


@dataclass
class ReadOptions:
    """How to read a source. ``None`` means detect automatically."""

    file_type: str | None = None  # "csv", "excel" or "paste"
    sheet: str | int | None = None
    header_row: int = 0  # 0-based row number of the header line
    separator: str | None = None
    decimal: str | None = None
    dayfirst: bool | None = None
    encoding: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_path(source: Any) -> bool:
    if isinstance(source, os.PathLike):
        return Path(source).exists()
    if not isinstance(source, str) or "\n" in source or "\t" in source:
        return False
    try:
        return Path(source).exists()
    except OSError:  # e.g. a long pasted string: "file name too long"
        return False


def _source_bytes(source: Any) -> tuple[bytes | None, str | None]:
    """(content, filename) for paths, bytes and file-like objects (e.g. Streamlit uploads)."""
    if _is_path(source):
        return Path(source).read_bytes(), Path(source).name
    if isinstance(source, bytes):
        return source, None
    if isinstance(source, str) and ("\n" in source or "\t" in source):
        return source.encode("utf-8"), None  # CSV text passed directly
    if hasattr(source, "getvalue"):
        return source.getvalue(), getattr(source, "name", None)
    if hasattr(source, "read"):
        return source.read(), getattr(source, "name", None)
    return None, None


def decode_text(content: bytes, encoding: str | None = None) -> tuple[str, str]:
    for enc in ([encoding] if encoding else ["utf-8-sig", "cp1252", "latin-1"]):
        try:
            return content.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file; set the encoding explicitly")


def sniff_separator(text: str, header_row: int = 0) -> str:
    """Choose the separator that splits the header line and the rows below it consistently.

    The header must contain the separator, so a decimal comma in a single-column table
    ('3,5') is not mistaken for one.
    """
    lines = [ln for ln in text.splitlines()[header_row:header_row + 30] if ln.strip()]
    best, best_score = None, 0.0
    for sep in SEPARATORS:
        counts = [len(next(csv.reader([ln], delimiter=sep))) - 1 for ln in lines]
        if not counts or counts[0] == 0:
            continue
        score = counts[0] * sum(c == counts[0] for c in counts) / len(counts)
        if score > best_score:
            best, best_score = sep, score
    if best is None:  # a single column: pick a separator that never occurs
        best = next((sep for sep in SEPARATORS if sep not in text), "\t")
    return best


def _rows_to_frame(rows: list[list[Any]], header_row: int, log: CleaningLog) -> pd.DataFrame:
    if header_row >= len(rows):
        raise ValueError(f"header row {header_row + 1} is beyond the end of the data ({len(rows)} rows)")
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        raise ValueError("the source contains no data")
    rows = [list(r) + [""] * (width - len(r)) for r in rows]
    if header_row:
        log.add("skip_rows", f"skipped {header_row} row(s) above the header", rows_affected=header_row)
    header = [str(h).strip() if not is_missing(h) else "" for h in rows[header_row]]
    names, seen = [], {}
    for i, h in enumerate(header):
        name = h or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    renamed = [f"{i + 1}: '{h}' → '{n}'" for i, (h, n) in enumerate(zip(header, names)) if h != n]
    if renamed:
        log.add("rename", "empty or duplicate headers renamed: " + "; ".join(renamed))
    body = pd.DataFrame(rows[header_row + 1:], columns=names, dtype=object)
    body = body.map(lambda v: "" if v is None else str(v)) if hasattr(body, "map") else body.applymap(
        lambda v: "" if v is None else str(v))
    empty_rows = body.apply(lambda r: all(is_missing(v) for v in r), axis=1)
    if empty_rows.any():
        log.add("drop_rows", "dropped empty rows", rows_affected=int(empty_rows.sum()))
        body = body.loc[~empty_rows]
    empty_cols = [c for c in body.columns if all(is_missing(v) for v in body[c])]
    if empty_cols and len(empty_cols) < body.shape[1]:
        log.add("drop_columns", f"dropped empty column(s): {', '.join(empty_cols)}")
        body = body.drop(columns=empty_cols)
    return body.reset_index(drop=True)


def read_csv_text(text: str, options: ReadOptions, log: CleaningLog) -> tuple[pd.DataFrame, dict[str, Any]]:
    sep = options.separator or sniff_separator(text, options.header_row)
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))  # blank lines kept: header_row counts them
    frame = _rows_to_frame(rows, options.header_row, log)
    info = {"separator": sep, "decimal_hint": "," if sep == ";" else "."}
    return frame, info


def list_sheets(source: Any) -> list[str]:
    content, _ = _source_bytes(source)
    if content is None:
        raise FileNotFoundError(source)
    with pd.ExcelFile(io.BytesIO(content)) as xls:
        return [str(s) for s in xls.sheet_names]


def read_excel_rows(content: bytes, options: ReadOptions, log: CleaningLog) -> tuple[pd.DataFrame, dict[str, Any]]:
    with pd.ExcelFile(io.BytesIO(content)) as xls:
        sheets = [str(s) for s in xls.sheet_names]
        sheet = options.sheet if options.sheet is not None else sheets[0]
        if isinstance(sheet, int):
            sheet = sheets[sheet]
        if sheet not in sheets:
            raise ValueError(f"No sheet {sheet!r}; sheets: {sheets}")
        grid = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)

    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, pd.Timestamp):
            return v.isoformat(sep=" ")
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    rows = [[cell(v) for v in row] for row in grid.itertuples(index=False)]
    frame = _rows_to_frame(rows, options.header_row, log)
    return frame, {"sheet": sheet, "sheets": sheets, "decimal_hint": "."}


def read_raw(source: Any, options: ReadOptions | None = None) -> tuple[pd.DataFrame, dict[str, Any], CleaningLog]:
    """Read a source into a raw text table. Returns (frame, info, log)."""
    options = options or ReadOptions()
    log = CleaningLog()
    file_type = options.file_type
    if file_type == "paste" or (file_type is None and isinstance(source, str) and not _is_path(source)
                                and ("\n" in source or "\t" in source)):
        text = source if isinstance(source, str) else source.decode()
        opts = ReadOptions(**{**options.to_dict(), "separator": options.separator
                              or ("\t" if "\t" in text else None)})
        frame, info = read_csv_text(text, opts, log)
        info.update({"file_type": "paste", "filename": None})
        log.add("read", f"pasted table, {len(frame)} rows × {frame.shape[1]} columns, "
                        f"separator {info['separator']!r}", rows_affected=len(frame))
        return frame, info, log

    content, filename = _source_bytes(source)
    if content is None:
        raise FileNotFoundError(f"Cannot read {source!r}")
    if file_type is None:
        suffix = Path(filename or "").suffix.lower()
        file_type = "excel" if suffix in EXCEL_SUFFIXES or content[:2] == b"PK" else "csv"
    if file_type == "excel":
        frame, info = read_excel_rows(content, options, log)
        info.update({"file_type": "excel", "filename": filename})
        log.add("read", f"Excel {filename or ''} sheet '{info['sheet']}', header row "
                        f"{options.header_row + 1}: {len(frame)} rows × {frame.shape[1]} columns",
                rows_affected=len(frame))
    else:
        text, enc = decode_text(content, options.encoding)
        frame, info = read_csv_text(text, options, log)
        info.update({"file_type": "csv", "filename": filename, "encoding": enc})
        log.add("read", f"CSV {filename or ''} ({enc}, separator {info['separator']!r}), header row "
                        f"{options.header_row + 1}: {len(frame)} rows × {frame.shape[1]} columns",
                rows_affected=len(frame))
    return frame, info, log


def load(
    source: Any,
    *,
    file_type: str | None = None,
    sheet: str | int | None = None,
    header_row: int = 0,
    separator: str | None = None,
    decimal: str | None = None,
    dayfirst: bool | None = None,
    encoding: str | None = None,
    name: str | None = None,
    time_unit: str = "min",
    currency: str = "EUR",
    suggest: bool = True,
) -> Dataset:
    """Read a CSV/Excel file, bytes, an upload or a pasted table into a typed Dataset.

    Column roles are suggested (``suggest=True``) but not confirmed; confirm them with
    ``analyser.io.mapping.apply_mapping``.
    """
    options = ReadOptions(file_type=file_type, sheet=sheet, header_row=header_row,
                          separator=separator, decimal=decimal, dayfirst=dayfirst, encoding=encoding)
    raw, info, log = read_raw(source, options)
    data, infos, lines = parse_frame(raw, decimal=decimal, decimal_hint=info.get("decimal_hint"),
                                     dayfirst=dayfirst)
    for col, line in zip(raw.columns, lines):
        log.add("parse", line, column=col)
    default_name = Path(info.get("filename") or "pasted table").stem
    ds = Dataset(
        data, name=name or default_name, columns=infos, raw=raw, log=log,
        source={**info, "options": {**options.to_dict(), "file_type": info["file_type"]}},
        time_unit=time_unit, currency=currency,
    )
    if suggest:
        from .mapping import suggest_roles

        for col, s in suggest_roles(ds).items():
            ds.columns[col].role = s.role
            ds.columns[col].role_reason = s.reason
            if s.amount_kind:
                ds.columns[col].amount_kind = s.amount_kind
            if s.unit:
                ds.columns[col].unit = s.unit
    return ds
