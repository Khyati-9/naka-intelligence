"""Join voluntary declarations to run-scoped person tracks; no visual inference."""
import json
import pandas as pd

KEYS = ["video", "analysis_id", "track_id"]


def track_roster(summary):
    rows = []
    for _, record in summary.iterrows():
        if pd.isna(record.get("analysis_id")):
            continue
        for track_id in json.loads(record.get("person_track_ids", "[]")):
            rows.append(dict(video=record["video"], analysis_id=record["analysis_id"],
                             track_id=int(track_id), gender="Unknown"))
    return pd.DataFrame(rows, columns=KEYS + ["gender"])


def validate_declarations(data, roster):
    if not set(KEYS + ["gender"]).issubset(data.columns):
        raise ValueError("CSV must contain video, analysis_id, track_id, gender.")
    data = data[KEYS + ["gender"]].copy()
    ids = pd.to_numeric(data.track_id, errors="coerce")
    if ids.isna().any() or (ids < 0).any() or (ids % 1 != 0).any():
        raise ValueError("Track IDs must be nonnegative whole numbers.")
    data["track_id"] = ids.astype(int)
    data["gender"] = data.gender.fillna("Unknown").astype(str).str.strip().str.lower().map(
        {"man": "Men", "men": "Men", "woman": "Women", "women": "Women", "unknown": "Unknown", "": "Unknown"})
    if data.gender.isna().any():
        raise ValueError("Use Men, Women, or Unknown; leave undeclared values Unknown.")
    if data.duplicated(KEYS).any():
        raise ValueError("Each video/run/track must appear only once.")
    checked = data.merge(roster[KEYS], on=KEYS, how="left", indicator=True)
    if (checked._merge != "both").any():
        raise ValueError("Some tracks belong to an old run or are not person tracks. Download the current roster.")
    return data


def count_declarations(summary, declarations):
    roster = track_roster(summary)
    matched = roster[KEYS].merge(declarations, on=KEYS, how="left")
    matched["gender"] = matched.gender.fillna("Unknown")
    rows = []
    for _, record in summary.iterrows():
        available = pd.notna(record.get("analysis_id"))
        counts = matched.loc[matched.video == record["video"], "gender"].value_counts()
        rows.append({"video": record["video"], "analysis_id": record.get("analysis_id"),
                     "status": "Complete" if available else "Reanalysis required",
                     **{label: int(counts.get(label, 0)) if available else None
                        for label in ["Men", "Women", "Unknown"]}})
    return pd.DataFrame(rows)
