"""Variable: one column of a Dataset plus its role in an analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANALYSIS_ROLES = ("response", "group", "block", "pair_id")


@dataclass
class Variable:
    """One column plus its roles.

    ``role`` is the import role (duration, amount, ...); ``analysis_role`` is its part in
    an analysis: response, group, block or pair ID. ``time_ordered_by`` names the time
    column when the rows are in time order (possible autocorrelation).
    """

    name: str
    values: pd.Series
    kind: str = "numeric"
    role: str | None = None
    analysis_role: str | None = "response"
    unit: str | None = None
    dataset: str | None = None
    time_ordered_by: str | None = None

    @property
    def n(self) -> int:
        """Number of non-missing values."""
        return int(self.values.notna().sum())

    @property
    def missing(self) -> int:
        return int(self.values.isna().sum())

    @property
    def label(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name

    def numeric(self) -> np.ndarray:
        """Non-missing values as a float array. Raises for non-numeric columns."""
        if self.kind != "numeric":
            raise TypeError(f"{self.name} is a {self.kind} column, not numeric")
        return self.values.dropna().to_numpy(dtype=float)
