"""
Combine TrackMan CSVs
---------------------
Merges several TrackMan game files into one CSV so the pitcher report can be run
on a stretch of games (e.g. the whole fall). Built for Google Colab: run it,
select all the game CSVs at once, and it downloads CSUN_Combined_TrackMan.csv.
"""

import pandas as pd
from google.colab import files

TEAM_CODE = "CAL_MAT"
OUTPUT_FILE = "CSUN_Combined_TrackMan.csv"


def read_trackman_csv(path):
    """Read one TrackMan CSV, skipping any metadata rows above the header row."""
    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    header_row = None
    for i, row in raw.iterrows():
        values = row.astype(str).str.strip()
        if (values == "PitcherTeam").any() and (values == "Pitcher").any():
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"{path}: could not find the TrackMan header row (Pitcher / PitcherTeam).")

    df = pd.read_csv(path, header=header_row, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    return df


uploaded = files.upload()
if len(uploaded) < 2:
    raise RuntimeError("Select at least two CSV files to combine.")

frames = []
for name, data in uploaded.items():
    with open(name, "wb") as f:
        f.write(data)
    game = read_trackman_csv(name).assign(SourceFile=name)
    print(f"{name}: {len(game)} rows")
    frames.append(game)

# Columns missing from one file are left blank for that file's rows
combined = pd.concat(frames, ignore_index=True, sort=False)

# Drop repeated pitches (e.g. the same game uploaded twice)
if "PitchUID" in combined.columns:
    before = len(combined)
    combined = combined.drop_duplicates(subset="PitchUID", keep="first")
    if before != len(combined):
        print(f"Removed {before - len(combined)} duplicate pitches (same PitchUID).")

# Put everything in chronological order
time_column = next((c for c in ["UTCDateTime", "LocalDateTime"] if c in combined.columns), None)
if time_column:
    combined = combined.sort_values(time_column, kind="stable")
combined = combined.reset_index(drop=True)

# Quick summary so you can confirm the right games went in
print("\nCombined rows:", len(combined))
if "Date" in combined:
    dates = pd.to_datetime(combined["Date"], errors="coerce").dropna()
    if len(dates):
        print(f"Date range: {dates.min():%m-%d-%Y} to {dates.max():%m-%d-%Y}  ({dates.nunique()} dates)")
if "BatterTeam" in combined:
    opponents = sorted(set(combined["BatterTeam"].dropna().astype(str)) - {TEAM_CODE})
    print("Opposing teams:", ", ".join(opponents))
if "PitcherTeam" in combined:
    print("CSUN pitches:", int((combined["PitcherTeam"].astype(str).str.strip() == TEAM_CODE).sum()))

combined.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {OUTPUT_FILE}")
files.download(OUTPUT_FILE)
