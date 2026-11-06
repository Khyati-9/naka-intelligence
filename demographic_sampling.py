"""Aggregate, anonymous demographic sampling helpers.

Rows contain counts observed by a researcher for a video/time interval. They
contain no track IDs, names, faces, or individual demographic records.
"""

from __future__ import annotations

import pandas as pd


COLUMNS = [
    "video", "timestamp_seconds", "men_count", "women_count", "unknown_gender_count",
    "age_0_17", "age_18_30", "age_31_50", "age_51_plus", "unknown_age_count",
    "confidence", "coverage",
]
COUNT_COLUMNS = [column for column in COLUMNS if column.endswith("count") or column.startswith("age_")]


def empty_samples() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def validate_samples(samples: pd.DataFrame, videos: set[str]) -> pd.DataFrame:
    missing = set(COLUMNS) - set(samples.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
    result = samples[COLUMNS].copy()
    if not result["video"].isin(videos).all():
        raise ValueError("Every video value must match an uploaded video filename.")
    for column in COUNT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or (result[column] < 0).any() or (result[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain nonnegative whole-number counts.")
        result[column] = result[column].astype(int)
    for column in ["confidence", "coverage"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or (result[column] < 0).any() or (result[column] > 1).any():
            raise ValueError(f"{column} must be between 0 and 1.")
    result["timestamp_seconds"] = pd.to_numeric(result["timestamp_seconds"], errors="coerce")
    if result["timestamp_seconds"].isna().any() or (result["timestamp_seconds"] < 0).any():
        raise ValueError("timestamp_seconds must be nonnegative.")
    return result


def summarize_samples(samples: pd.DataFrame, videos: set[str]) -> pd.DataFrame:
    result = pd.DataFrame({"video": sorted(videos)})
    if samples.empty:
        for column in ["men_count", "women_count", "unknown_gender_count", "age_0_17", "age_18_30", "age_31_50", "age_51_plus", "unknown_age_count"]:
            result[column] = pd.NA
        result["confidence"] = pd.NA
        result["coverage"] = pd.NA
        return result
    grouped = samples.groupby("video", as_index=False).agg(
        **{column: (column, "sum") for column in COUNT_COLUMNS},
        confidence=("confidence", "mean"), coverage=("coverage", "mean"),
    )
    return result.merge(grouped, on="video", how="left")
