"""Import profiles: save how a file was read and mapped, so the same export loads in one click.

Profiles are JSON files in ``~/.data-analyser/profiles`` (override with the
ANALYSER_HOME environment variable or the ``directory`` argument).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.dataset import Dataset
from .derive import derive
from .load import load
from .mapping import ColumnMapping, apply_mapping


def analyser_home() -> Path:
    return Path(os.environ.get("ANALYSER_HOME", Path.home() / ".data-analyser"))


def profile_dir(directory: str | Path | None = None) -> Path:
    path = Path(directory) if directory else analyser_home() / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    if not slug:
        raise ValueError("profile name must contain letters or digits")
    return slug


@dataclass
class ImportProfile:
    name: str
    file_type: str
    mapping: ColumnMapping
    columns: list[str]
    sheet: str | int | None = None
    header_row: int = 0
    separator: str | None = None
    decimal: str | None = None
    dayfirst: bool | None = None
    encoding: str | None = None
    time_unit: str = "min"
    currency: str = "EUR"
    derivations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _dt.datetime.now().astimezone().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mapping"] = self.mapping.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImportProfile":
        d = dict(d)
        d["mapping"] = ColumnMapping.from_dict(d["mapping"])
        return cls(**d)

    def matches(self, columns: list[str]) -> bool:
        """True if every column this profile maps is present."""
        return set(self.mapping.roles) <= set(columns)


def profile_from_dataset(ds: Dataset, name: str, derivations: list[str] | None = None) -> ImportProfile:
    if ds.mapping is None:
        raise ValueError("confirm the column mapping before saving a profile")
    opts = ds.source.get("options", {})
    keys = derivations
    if keys is None:
        keys = sorted({d.split(":")[0] for d in ds.derivations if d.split(":")[0] in
                       ("interarrival", "duration", "waiting", "profit")})
    return ImportProfile(
        name=name, file_type=ds.source.get("file_type", "csv"), mapping=ds.mapping,
        columns=list(ds.raw.columns) if ds.raw is not None else list(ds.mapping.roles),
        sheet=ds.source.get("sheet", opts.get("sheet")), header_row=opts.get("header_row", 0),
        separator=ds.source.get("separator"), decimal=opts.get("decimal"),
        dayfirst=opts.get("dayfirst"), encoding=opts.get("encoding"),
        time_unit=ds.time_unit, currency=ds.currency, derivations=keys,
    )


def save_profile(profile: ImportProfile, directory: str | Path | None = None) -> Path:
    path = profile_dir(directory) / f"{_slug(profile.name)}.json"
    path.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_profile(name: str, directory: str | Path | None = None) -> ImportProfile:
    path = profile_dir(directory) / f"{_slug(name)}.json"
    if not path.exists():
        raise FileNotFoundError(f"No import profile named {name!r} in {path.parent}")
    return ImportProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_profiles(directory: str | Path | None = None) -> list[ImportProfile]:
    out = []
    for path in sorted(profile_dir(directory).glob("*.json")):
        try:
            out.append(ImportProfile.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, KeyError, TypeError):
            continue  # skip unreadable files
    return out


def delete_profile(name: str, directory: str | Path | None = None) -> None:
    path = profile_dir(directory) / f"{_slug(name)}.json"
    path.unlink(missing_ok=True)


def matching_profiles(columns: list[str], directory: str | Path | None = None) -> list[ImportProfile]:
    return [p for p in list_profiles(directory) if p.matches(columns)]


def apply_profile(source: Any, profile: ImportProfile, *, name: str | None = None) -> Dataset:
    """Read ``source`` with the profile's settings, apply its mapping and run its derivations."""
    ds = load(source, file_type=profile.file_type if profile.file_type != "paste" else "paste",
              sheet=profile.sheet, header_row=profile.header_row, separator=profile.separator,
              decimal=profile.decimal, dayfirst=profile.dayfirst, encoding=profile.encoding,
              name=name, time_unit=profile.time_unit, currency=profile.currency)
    missing = [c for c in profile.mapping.roles if c not in ds.data.columns]
    if missing:
        raise ValueError(f"profile '{profile.name}' expects column(s) {missing}, not found in this file")
    ds.log.add("profile", f"applied import profile '{profile.name}'")
    ds = apply_mapping(ds, profile.mapping, inplace=True)
    if profile.derivations:
        derive(ds, profile.derivations)
    return ds
