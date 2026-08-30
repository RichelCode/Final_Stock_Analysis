"""
Posterior convergence diagnostics.

Wraps arviz so the rest of the project depends on one small interface rather
than on an arviz version's own API, which changed shape between 0.x and 1.x.

What the numbers mean
---------------------
R-hat compares variance between chains to variance within them. Values near 1
mean the chains have forgotten where they started and are exploring the same
distribution; the usual threshold is 1.01.

Effective sample size counts how many independent draws the correlated chain is
worth. Bulk ESS governs the reliability of posterior means, tail ESS the
reliability of the interval endpoints this project actually reports. Around 400
per quantity is the common minimum.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400.0


def convergence_table(by_chain: dict[str, np.ndarray]) -> pd.DataFrame:
    """Summarise draws shaped (chain, draw, ...) for each named parameter."""
    idata = az.from_dict({"posterior": by_chain})
    summary = az.summary(idata)
    summary.index.name = "parameter"
    return summary.reset_index()


def convergence_report(table: pd.DataFrame) -> dict:
    """Reduce a convergence table to a pass/fail verdict."""
    rhat = pd.to_numeric(table["r_hat"], errors="coerce")
    ess_bulk = pd.to_numeric(table["ess_bulk"], errors="coerce")
    ess_tail = pd.to_numeric(table["ess_tail"], errors="coerce")

    worst_rhat_row = table.loc[rhat.idxmax()] if rhat.notna().any() else None
    worst_ess_row = table.loc[ess_bulk.idxmin()] if ess_bulk.notna().any() else None

    return {
        "max_rhat": float(rhat.max()),
        "max_rhat_parameter": None if worst_rhat_row is None else worst_rhat_row["parameter"],
        "min_ess_bulk": float(ess_bulk.min()),
        "min_ess_bulk_parameter": None if worst_ess_row is None else worst_ess_row["parameter"],
        "min_ess_tail": float(ess_tail.min()),
        "n_parameters": len(table),
        "rhat_ok": bool(rhat.max() < RHAT_THRESHOLD),
        "ess_ok": bool(ess_bulk.min() > ESS_THRESHOLD and ess_tail.min() > ESS_THRESHOLD),
    }


def print_report(report: dict) -> None:
    rhat_mark = "OK" if report["rhat_ok"] else "FAIL"
    ess_mark = "OK" if report["ess_ok"] else "FAIL"
    print(f"  max R-hat    {report['max_rhat']:.4f}  "
          f"({report['max_rhat_parameter']})  threshold < {RHAT_THRESHOLD}  [{rhat_mark}]")
    print(f"  min ESS bulk {report['min_ess_bulk']:.0f}  "
          f"({report['min_ess_bulk_parameter']})  threshold > {ESS_THRESHOLD:.0f}  [{ess_mark}]")
    print(f"  min ESS tail {report['min_ess_tail']:.0f}")
    print(f"  parameters monitored: {report['n_parameters']}")
