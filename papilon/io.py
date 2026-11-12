#   Copyright 2024 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Getting data in: warehouse loaders and checks before a model is fitted.

The loaders are thin and optional. ``read_sql`` works with any database pandas can
read (a SQLAlchemy engine or a DB-API connection, including the Snowflake, BigQuery,
Redshift, Databricks and Postgres drivers). ``from_bigquery`` and ``from_snowflake``
import their client libraries only when called, so Papilon itself needs neither.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["read_sql", "from_bigquery", "from_snowflake", "validate_frame", "long_to_wide"]


def read_sql(query: str, con, date_col: str | None = None, **kwargs) -> pd.DataFrame:
    """Run ``query`` on ``con`` (SQLAlchemy engine or DB-API connection) and parse ``date_col``."""
    df = pd.read_sql(query, con, **kwargs)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def from_bigquery(query: str, project: str | None = None, date_col: str | None = None) -> pd.DataFrame:
    """Run ``query`` in BigQuery with application default credentials. Needs ``google-cloud-bigquery``."""
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise ImportError('pip install "google-cloud-bigquery[pandas]" to read from BigQuery') from e
    df = bigquery.Client(project=project).query(query).to_dataframe()
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def from_snowflake(query: str, date_col: str | None = None, **connect_kwargs) -> pd.DataFrame:
    """Run ``query`` in Snowflake. ``connect_kwargs`` go to ``snowflake.connector.connect``
    (account, user, authenticator, warehouse, database, schema). Needs ``snowflake-connector-python[pandas]``."""
    try:
        import snowflake.connector
    except ImportError as e:
        raise ImportError('pip install "snowflake-connector-python[pandas]" to read from Snowflake') from e
    with snowflake.connector.connect(**connect_kwargs) as conn:
        df = conn.cursor().execute(query).fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]
    if date_col:
        df[date_col.lower()] = pd.to_datetime(df[date_col.lower()])
    return df


def long_to_wide(df: pd.DataFrame, date_col: str, channel_col: str, value_col: str,
                 index_cols: list[str] | None = None) -> pd.DataFrame:
    """Pivot spend stored one row per date and channel (as warehouses often hold it) to one column per channel."""
    idx = [date_col] + list(index_cols or [])
    out = df.pivot_table(index=idx, columns=channel_col, values=value_col, aggfunc="sum", fill_value=0.0)
    out.columns = [str(c) for c in out.columns]
    return out.reset_index()


def validate_frame(df: pd.DataFrame, target: str, media: list[str], date_col: str,
                   controls: list[str] | None = None, geo_col: str | None = None) -> pd.DataFrame:
    """Check a frame before fitting. Returns one row per problem, most serious first; empty means none found.

    Errors stop a fit from being meaningful (missing columns, missing or negative
    values, duplicate periods). Warnings are worth reading (gaps in the calendar,
    channels that never vary or rarely run, fewer than two years of history,
    extreme spikes in the target).
    """
    rows = []
    add = lambda sev, check, column, detail: rows.append(
        {"severity": sev, "check": check, "column": column, "detail": detail})
    controls = list(controls or [])
    cols = [date_col, target] + list(media) + controls + ([geo_col] if geo_col else [])
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        add("error", "missing_column", c, "not in the data")
    if missing:
        return pd.DataFrame(rows)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        add("error", "bad_dates", date_col, f"{int(dates.isna().sum())} values are not dates")
    groups = [g for _, g in df.assign(_d=dates).groupby(geo_col)] if geo_col else [df.assign(_d=dates)]
    for g in groups:
        where = f" in {g[geo_col].iloc[0]}" if geo_col else ""
        if g["_d"].duplicated().any():
            add("error", "duplicate_periods", date_col, f"{int(g['_d'].duplicated().sum())} repeated dates{where}")
        d = g["_d"].sort_values().diff().dropna()
        if len(d) and (d > 1.5 * d.median()).any():
            add("warning", "calendar_gaps", date_col, f"{int((d > 1.5 * d.median()).sum())} gaps longer than one period{where}")
        if len(g) < 104:
            add("warning", "short_history", date_col, f"{len(g)} periods{where}; two years or more separates season from media")
    for c in [target] + list(media) + controls:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            add("error", "missing_values", c, f"{int(v.isna().sum())} missing or non-numeric values")
        if c in media and (v < 0).any():
            add("error", "negative_spend", c, f"{int((v < 0).sum())} negative values")
        if v.std() == 0:
            add("warning", "constant", c, "never varies, so its effect cannot be estimated")
        if c in media and (v > 0).mean() < 0.2:
            add("warning", "rarely_active", c, f"active in {(v > 0).mean():.0%} of periods")
    y = pd.to_numeric(df[target], errors="coerce")
    z = (y - y.median()).abs() / (1.4826 * (y - y.median()).abs().median() or 1.0)
    if (z > 6).any():
        add("warning", "target_spikes", target, f"{int((z > 6).sum())} periods far from the rest; check for data errors or events to control for")
    if len(media) > 1:
        corr = df[list(media)].corr().abs()
        for i, a in enumerate(media):
            for b in media[i + 1:]:
                if corr.loc[a, b] > 0.9:
                    add("warning", "collinear_channels", f"{a}, {b}",
                        f"spend correlation {corr.loc[a, b]:.2f}; the model will struggle to separate them without a lift test or geo data")
    order = {"error": 0, "warning": 1}
    out = pd.DataFrame(rows, columns=["severity", "check", "column", "detail"])
    return out.sort_values("severity", key=lambda s: s.map(order), kind="stable", ignore_index=True)
