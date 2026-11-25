# Pitcher Reports

Automated one-page reports for every pitcher on the staff, built from TrackMan data after each outing or across a stretch of games.

![Sample pitcher report](images/sample-report.png)

Each page shows:
- **Hitter Performance:** BAA, exit velocity, batted-ball rates and swing / whiff / chase rates, overall and by batter side
- **Pitch Location vs RHH / LHH:** with a heart / shadow / chase / waste breakdown by pitch type
- **Release Slot:** drawn on a silhouette scaled to the pitcher's height and arm slot
- **Pitch Movement:** compared with D1 averages for each pitch type and handedness
- **Pitch Characteristics:** velocity, spin, movement, release, extension and approach angle

## Files

| File | What it does |
|---|---|
| `pitcher_report.py` | Builds the PDF: one page per pitcher in the TrackMan file |
| `combine_trackman.py` | Merges several game files into one CSV for cumulative reports |

## How to run

Both scripts are written for Google Colab:

1. Open a new Colab notebook and paste in the script.
2. Run the cell and upload a TrackMan CSV when prompted (for `combine_trackman.py`, select all the game files at once).
3. The PDF (or combined CSV) downloads when it finishes.

If a pitcher isn't in `CSUN_ROSTER`, the script asks for his arm slot and height.

**Requires:** pandas, numpy, matplotlib, Pillow (all preinstalled in Colab)

*Player names in this public version have been replaced with placeholders. No team data is included.*
