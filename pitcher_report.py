"""
CSUN Pitcher Report
-------------------
Builds a landscape one-page report for every CSUN pitcher in a TrackMan file:
results against hitters, pitch locations vs. righties and lefties, a location
breakdown by pitch type, release slot drawn on a body silhouette, pitch movement
compared with D1 averages, and a pitch characteristics table.

Works on one game or a combined file (see combine_trackman.py). Built for Google
Colab: run it, upload the CSV, and the PDF downloads. The logo, arm-slot
silhouettes and D1 movement averages are embedded, so nothing else is needed.
"""

import base64
import difflib
import io
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.colab import files
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from PIL import Image


# ============================================================
# Team settings
# ============================================================

TEAM_CODE = "CAL_MAT"

# Throwing hand, arm slot (High / Low) and height in inches for each pitcher.
# Pitchers missing from this list are asked about when the script runs.
CSUN_ROSTER = {
    # Player names replaced for the public version of this script.
    "Pitcher, A":         {"hand": "R", "slot": "High", "height": 71},
    "Pitcher, B":         {"hand": "L", "slot": "Low",  "height": 76},
    "Pitcher, C":         {"hand": "R", "slot": "Low",  "height": 73},
    "Pitcher, D":         {"hand": "R", "slot": "High", "height": 74},
    "Pitcher, E":         {"hand": "R", "slot": "Low",  "height": 75},
    "Pitcher, F":         {"hand": "R", "slot": "High", "height": 71},
    "Pitcher, G":         {"hand": "R", "slot": "High", "height": 77},
    "Pitcher, H":         {"hand": "R", "slot": "High", "height": 77},
    "Pitcher, I":         {"hand": "R", "slot": "Low",  "height": 78},
    "Pitcher, J":         {"hand": "L", "slot": "High", "height": 75},
    "Pitcher, K":         {"hand": "R", "slot": "Low",  "height": 73},
    "Pitcher, L":         {"hand": "R", "slot": "Low",  "height": 69},
    "Pitcher, M":         {"hand": "R", "slot": "High", "height": 74},
    "Pitcher, N":         {"hand": "L", "slot": "Low",  "height": 72},
    "Pitcher, O":         {"hand": "L", "slot": "Low",  "height": 75},
    "Pitcher, P":         {"hand": "R", "slot": "Low",  "height": 71},
    "Pitcher, Q":         {"hand": "R", "slot": "High", "height": 72},
    "Pitcher, R":         {"hand": "R", "slot": "Low",  "height": 76},
    "Pitcher, S":         {"hand": "R", "slot": "Low",  "height": 75},
    "Pitcher, T":         {"hand": "R", "slot": "Low",  "height": 69},
    "Pitcher, U":         {"hand": "R", "slot": "Low",  "height": 75},
}


# ============================================================
# Pitch types, colors and D1 benchmarks
# ============================================================

PITCH_COLORS = {
    "Fastball": "#FF69B4", "Sinker": "turquoise", "Cutter": "purple",
    "ChangeUp": "blue", "Slider": "#FFDF00", "Sweeper": "#00BFFF",
    "Curveball": "green", "Splitter": "brown", "Other": "gray",
}
PITCH_ORDER = ["Fastball", "Sinker", "Cutter", "Splitter",
               "ChangeUp", "Slider", "Sweeper", "Curveball", "Other"]

# Average induced vertical break / horizontal break (inches) for D1 pitchers
D1_MOVEMENT_AVERAGES = {
    "R": {
        "Fastball":  {"IVB": 16.54,  "HB": 9.90},
        "Sinker":    {"IVB": 8.66,   "HB": 16.75},
        "Cutter":    {"IVB": 5.69,   "HB": -2.13},
        "ChangeUp":  {"IVB": 6.16,   "HB": 14.15},
        "Slider":    {"IVB": 0.40,   "HB": -6.45},
        "Sweeper":   {"IVB": -0.58,  "HB": -13.84},
        "Curveball": {"IVB": -10.80, "HB": -8.36},
        "Splitter":  {"IVB": 4.92,   "HB": 9.14},
    },
    "L": {
        "Fastball":  {"IVB": 16.33,  "HB": -10.66},
        "Sinker":    {"IVB": 9.72,   "HB": -15.46},
        "Cutter":    {"IVB": 6.06,   "HB": 1.56},
        "ChangeUp":  {"IVB": 6.81,   "HB": -12.96},
        "Slider":    {"IVB": 0.09,   "HB": 7.49},
        "Sweeper":   {"IVB": -0.34,  "HB": 13.13},
        "Curveball": {"IVB": -10.48, "HB": 10.52},
        "Splitter":  {"IVB": 5.20,   "HB": -8.21},
    },
}

STRIKE_CALLS = {"StrikeCalled", "StrikeSwinging", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"}
SWING_CALLS = {"StrikeSwinging", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay", "FoulBall"}
HIT_RESULTS = {"Single", "Double", "Triple", "HomeRun"}
AB_RESULTS = HIT_RESULTS | {"Out", "FieldersChoice", "Error", "Strikeout"}
NON_AB_RESULTS = {"Walk", "HitByPitch", "CatcherInterference", "Sacrifice"}


# ============================================================
# Location zones (inches, pitcher's view)
# ============================================================
# Heart = inside the strike zone, Shadow = the band just outside it, Chase = the
# next band out, Waste = everything beyond. These are location bands, not
# called-strike results. PlateLocSide is used as-is (no flip).

ZONE_X = 8.5                          # half-width of the zone
ZONE_BOTTOM, ZONE_TOP = 18.0, 42.0
SHADOW_X, SHADOW_LOW, SHADOW_HIGH = 3.5, 3.0, 3.0
CHASE_X, CHASE_LOW, CHASE_HIGH = 8.0, 6.0, 6.0

ZONE_COLUMNS = [("HeartZone", "HeartZone%"), ("ShadowZone", "ShadowZone%"),
                ("ChaseZone", "ChaseZone%"), ("WasteZone", "WasteZone%")]

MAX_ZONE_PITCHES = 60     # max dots per location plot; extra pitches are sampled by usage
ZONE_SAMPLE_SEED = 42     # fixed seed so re-running a file gives the same plot


# ============================================================
# Page layout (inches; "top" = distance down from the top of the page)
# ============================================================

FIG_W, FIG_H = 11.0, 8.5
MARGIN_L = 0.45
CELL_W = 0.80
CELL_H = 0.21
LEFT_COL_W = 8 * CELL_W              # width of the Hitter Performance table and zone area
FULL_W = FIG_W - 2 * MARGIN_L

LOGO_WIDTH_IN = 1.25
LOGO_TOP_IN = 0.14
LOGO_ASPECT = 0.4375                 # height / width of the embedded logo

# Release-slot silhouette: the top of the head sits at 66 in for a 6'2" pitcher
# and scales with the pitcher's height.
SILHOUETTE_TOP_IN = 66.0
SILHOUETTE_REF_HEIGHT = 74.0
SCALE_SILHOUETTE_BY_HEIGHT = True


# ============================================================
# Small helpers
# ============================================================

def clean_text(s):
    return str(s).strip() if pd.notna(s) else ""


def to_number(series):
    return pd.to_numeric(series, errors="coerce")


def fmt_pct(numerator, denominator):
    return "—" if not denominator else f"{100 * numerator / denominator:.1f}%"


def fmt_avg(values, digits=1):
    values = to_number(pd.Series(values)).dropna()
    return "—" if values.empty else f"{values.mean():.{digits}f}"


def fmt_max(values, digits=1):
    values = to_number(pd.Series(values)).dropna()
    return "—" if values.empty else f"{values.max():.{digits}f}"


def normalize_hand(value):
    s = clean_text(value).lower()
    if s.startswith("r"):
        return "R"
    if s.startswith("l"):
        return "L"
    return ""


def name_key(name):
    """Name for matching: ignores extra spaces, non-breaking spaces and capitalization."""
    return re.sub(r"\s+", " ", str(name).replace("\xa0", " ")).strip().casefold()


# ============================================================
# Reading the TrackMan file
# ============================================================

def read_trackman_csv(path):
    """Read a TrackMan CSV, skipping metadata rows above the header.
    Returns (data, metadata rows)."""
    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    header_row = None
    for i, row in raw.iterrows():
        values = row.astype(str).str.strip()
        if (values == "PitcherTeam").any() and (values == "Pitcher").any():
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not locate the TrackMan header row (Pitcher and PitcherTeam).")
    df = pd.read_csv(path, header=header_row, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    return df, raw.iloc[:header_row]


def get_report_date(df, metadata):
    """Game date from the metadata rows (YYYYMMDD), else the data, else today."""
    for value in metadata.to_numpy().ravel():
        value = clean_text(value)
        if re.fullmatch(r"\d{8}", value):
            return pd.to_datetime(value, format="%Y%m%d").strftime("%m-%d-%Y")
    for col in ["Date", "LocalDateTime", "UTCDate"]:
        if col in df and df[col].notna().any():
            dates = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(dates):
                return dates.iloc[0].strftime("%m-%d-%Y")
    return datetime.today().strftime("%m-%d-%Y")


def summarize_games(df, metadata):
    """Describe what's in the file. One game gives a single date and opponent;
    several give a date range and an opponent list.
    Returns (date_label, opponent_label, opponents, dates, n_games)."""
    dates = pd.Series(dtype="datetime64[ns]")
    for col in ["Date", "UTCDate", "LocalDateTime"]:
        if col in df and df[col].notna().any():
            d = pd.to_datetime(df[col], errors="coerce", utc=False)
            try:
                d = d.dt.tz_localize(None)
            except Exception:
                pass
            d = d.dt.normalize().dropna()
            if len(d):
                dates = d
                break

    opponents = sorted(set(df["BatterTeam"].dropna().astype(str).str.strip())) if "BatterTeam" in df else []

    if len(dates) == 0:
        date_label, date_list = get_report_date(df, metadata), []
    else:
        date_list = sorted(dates.unique())
        first, last = pd.Timestamp(date_list[0]), pd.Timestamp(date_list[-1])
        date_label = first.strftime("%m-%d-%Y") if first == last else f"{first:%m-%d-%Y} to {last:%m-%d-%Y}"

    n_games = df["GameID"].nunique() if "GameID" in df else max(len(date_list), 1)

    if len(opponents) == 0:
        opponent_label = "Opponent"
    elif len(opponents) <= 3:
        opponent_label = " / ".join(opponents)
    else:
        opponent_label = f"{len(opponents)} opponents"
    return date_label, opponent_label, opponents, date_list, n_games


# ============================================================
# Stats and tables
# ============================================================

def add_location_zones(g):
    """Add location in inches and which zone band each pitch landed in."""
    side = to_number(g["PlateLocSide"]) * 12
    height = to_number(g["PlateLocHeight"]) * 12
    abs_side = side.abs()

    in_heart = (abs_side <= ZONE_X) & height.between(ZONE_BOTTOM, ZONE_TOP)
    in_shadow = ((abs_side <= ZONE_X + SHADOW_X)
                 & height.between(ZONE_BOTTOM - SHADOW_LOW, ZONE_TOP + SHADOW_HIGH)
                 & ~in_heart)
    in_chase = ((abs_side <= ZONE_X + SHADOW_X + CHASE_X)
                & height.between(ZONE_BOTTOM - SHADOW_LOW - CHASE_LOW, ZONE_TOP + SHADOW_HIGH + CHASE_HIGH)
                & ~in_heart & ~in_shadow)

    g = g.copy()
    g["LocSideIn"] = side
    g["LocHeightIn"] = height
    g["LocationZone"] = np.select([in_heart, in_shadow, in_chase],
                                  ["HeartZone", "ShadowZone", "ChaseZone"], default="WasteZone")
    return g


def pitch_types_by_usage(g):
    """Pitch types, most-thrown first (ties fall back to PITCH_ORDER)."""
    counts = g["TaggedPitchType"].value_counts()
    rank = {p: i for i, p in enumerate(PITCH_ORDER)}
    return sorted(counts.index, key=lambda p: (-counts[p], rank.get(p, 99)))


def sample_zone_pitches(sub, cap=MAX_ZONE_PITCHES, seed=ZONE_SAMPLE_SEED):
    """If there are more than `cap` located pitches, pick `cap` of them so each pitch
    type keeps its share of usage (largest-remainder rounding so the total is exactly
    `cap`). Returns (pitches to plot, total located)."""
    sub = sub[sub["LocSideIn"].notna() & sub["LocHeightIn"].notna()]
    n = len(sub)
    if n <= cap:
        return sub, n

    counts = sub["TaggedPitchType"].value_counts()
    exact = counts / n * cap
    quota = np.floor(exact).astype(int)
    leftover = cap - int(quota.sum())
    for pitch in (exact - quota).sort_values(ascending=False, kind="stable").index[:leftover]:
        quota[pitch] += 1

    parts = [sub[sub["TaggedPitchType"].eq(p)].sample(n=int(q), random_state=seed)
             for p, q in quota.items() if q > 0]
    return pd.concat(parts), n


def plate_appearance_events(g):
    """One row per plate appearance: the pitch carrying the result.
    PlayResult/KorBB are event fields, so this keeps multi-pitch PAs from
    being counted more than once."""
    x = g.copy()
    pa_keys = [c for c in ["GameID", "Inning", "Top/Bottom", "PAofInning", "BatterId", "PitcherId"]
               if c in x.columns]
    if not pa_keys:
        x["_PAKey"] = np.arange(len(x))
        pa_keys = ["_PAKey"]

    x["_event"] = (x["PlayResult"].isin(AB_RESULTS | NON_AB_RESULTS)
                   | x["KorBB"].isin({"Walk", "Strikeout", "HitByPitch"}))
    events = x[x["_event"]].copy()
    if events.empty:
        return events
    return events.sort_values("PitchNo").drop_duplicates(pa_keys, keep="last")


def hitter_summary(g):
    """How hitters did against this group of pitches."""
    pa = plate_appearance_events(g)
    hits = pa["PlayResult"].isin(HIT_RESULTS).sum() if len(pa) else 0

    # TrackMan records strikeouts in KorBB while PlayResult stays "Undefined",
    # so strikeouts have to be counted as at-bats from KorBB.
    if len(pa):
        is_k = pa["KorBB"].eq("Strikeout")
        is_ab = ((pa["PlayResult"].isin(AB_RESULTS - {"Strikeout"}) | is_k)
                 & ~pa["PlayResult"].isin(NON_AB_RESULTS)
                 & ~pa["KorBB"].isin({"Walk", "HitByPitch"}))
        at_bats = int(is_ab.sum())
    else:
        at_bats = 0

    in_play = g["PitchCall"].eq("InPlay")
    batted = g[in_play]
    batted = batted[~batted["TaggedHitType"].eq("Bunt")]
    exit_velo = to_number(batted["ExitSpeed"]).dropna()

    hit_types = g.loc[in_play, "TaggedHitType"]
    ground_balls = hit_types.eq("GroundBall").sum()
    fly_and_pop = hit_types.isin(["FlyBall", "Popup", "PopUp"]).sum()

    swings = g["PitchCall"].isin(SWING_CALLS)
    outside = g["LocationZone"].ne("HeartZone") & g["LocationZone"].ne("ShadowZone")

    return {
        "BAA": "—" if not at_bats else f"{hits / at_bats:.3f}",
        "AvgEV": "—" if exit_velo.empty else f"{exit_velo.mean():.1f}",
        "GB%": fmt_pct(ground_balls, len(hit_types)),
        "FB+PU%": fmt_pct(fly_and_pop, len(hit_types)),
        "Swing%": fmt_pct(swings.sum(), len(g)),
        "Whiff%": fmt_pct(g["PitchCall"].eq("StrikeSwinging").sum(), swings.sum()),
        "Chase%": fmt_pct((swings & outside).sum(), outside.sum()),
    }


def pitch_location_table(g, pitch_types):
    """Strike% and zone-band percentages for each pitch type, plus a total row."""
    def row_for(label, x):
        n = len(x)
        row = {"Pitch": label, "Strike%": fmt_pct(x["PitchCall"].isin(STRIKE_CALLS).sum(), n)}
        for zone, column in ZONE_COLUMNS:
            row[column] = fmt_pct(x["LocationZone"].eq(zone).sum(), n)
        return row

    rows = [row_for(p, g[g["TaggedPitchType"].eq(p)]) for p in pitch_types]
    rows.append(row_for("Total", g))
    return pd.DataFrame(rows)


def spin_clock(degrees):
    """Spin axis in degrees -> clock face, to the nearest half hour."""
    if pd.isna(degrees):
        return "—"
    hour = (float(degrees) % 360) / 30
    half_hours = int(round(hour * 2)) % 24
    return f"{(half_hours // 2 or 12)}:{'30' if half_hours % 2 else '00'}"


def pitch_characteristics(g, pitch_types):
    # RelHeight / RelSide / Extension come from TrackMan in feet.
    # Stuff+ is a placeholder until the Stuff+ model is finished.
    rows = []
    for p in pitch_types:
        x = g[g["TaggedPitchType"].eq(p)]
        rows.append([
            p, len(x), fmt_pct(len(x), len(g)),
            fmt_avg(x["RelSpeed"]), fmt_max(x["RelSpeed"]),
            fmt_avg(x["SpinRate"], 0), spin_clock(to_number(x["SpinAxis"]).mean()),
            fmt_avg(x["InducedVertBreak"]), fmt_avg(x["HorzBreak"]),
            fmt_avg(x["RelHeight"]), fmt_avg(x["RelSide"]), fmt_avg(x["Extension"]),
            fmt_avg(x["VertApprAngle"]), "—",
        ])
    return pd.DataFrame(rows, columns=["Pitch", "#", "Usage%", "Avg Velo", "Max Velo", "Spin", "Spin Dir",
                                       "IVB", "HB", "Rel H (ft)", "Rel S (ft)", "Ext (ft)", "VAA", "Stuff+"])


# ============================================================
# Drawing helpers
# ============================================================

def add_axes_in(fig, left, top, width, height, **kw):
    """Add an axes positioned in inches from the left / top of the page."""
    return fig.add_axes([left / FIG_W, 1 - (top + height) / FIG_H, width / FIG_W, height / FIG_H], **kw)


def label_in(fig, left, top, text, **kw):
    kw.setdefault("fontsize", 10)
    kw.setdefault("fontweight", "bold")
    fig.text(left / FIG_W, 1 - top / FIG_H, text, va="top", ha=kw.pop("ha", "left"), **kw)


def draw_table(fig, df, left, top, width, row_h, fontsize=8, header_fontsize=None, red_headers=()):
    """Draw a table in an axes of exact size so every table on the page shares
    the same row height."""
    header_fontsize = header_fontsize or fontsize
    ax = add_axes_in(fig, left, top, width, (len(df) + 1) * row_h)
    ax.axis("off")
    if df.empty:
        ax.text(.5, .5, "No data available", ha="center", va="center")
        return ax

    t = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns),
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=header_fontsize)
            cell.set_facecolor("#D62828" if df.columns[c] in red_headers else "#333333")
        else:
            cell.set_text_props(fontsize=fontsize)
            if r % 2 == 0:
                cell.set_facecolor("#F1F3F5")
    return ax


def plate_art(ax):
    # Home plate seen from the pitcher: flat edge at the bottom, point toward the catcher
    plate = Polygon([(-8.5, 5.0), (8.5, 5.0), (8.5, 7.0), (0, 11.0), (-8.5, 7.0)],
                    closed=True, facecolor="#F3F3F3", edgecolor="black",
                    linewidth=1.4, zorder=1, clip_on=False)
    ax.add_patch(plate)


def draw_zone(ax):
    """Strike zone with a 3x3 grid, plus dashed shadow and dotted chase boundaries."""
    ax.add_patch(Rectangle((-ZONE_X, ZONE_BOTTOM), 2 * ZONE_X, ZONE_TOP - ZONE_BOTTOM,
                           fill=False, edgecolor="black", linewidth=1.7, zorder=3))
    for x in [-ZONE_X / 3, ZONE_X / 3]:
        ax.plot([x, x], [ZONE_BOTTOM, ZONE_TOP], color="gray", lw=.55, alpha=.7)
    for y in [ZONE_BOTTOM + 8, ZONE_BOTTOM + 16]:
        ax.plot([-ZONE_X, ZONE_X], [y, y], color="gray", lw=.55, alpha=.7)

    shadow_w = ZONE_X + SHADOW_X
    ax.add_patch(Rectangle((-shadow_w, ZONE_BOTTOM - SHADOW_LOW), 2 * shadow_w,
                           ZONE_TOP - ZONE_BOTTOM + SHADOW_LOW + SHADOW_HIGH,
                           fill=False, edgecolor="#777777", linestyle="--", linewidth=.8, zorder=2))
    chase_w = ZONE_X + SHADOW_X + CHASE_X
    ax.add_patch(Rectangle((-chase_w, ZONE_BOTTOM - SHADOW_LOW - CHASE_LOW), 2 * chase_w,
                           ZONE_TOP - ZONE_BOTTOM + SHADOW_LOW + SHADOW_HIGH + CHASE_LOW + CHASE_HIGH,
                           fill=False, edgecolor="#BBBBBB", linestyle=":", linewidth=.8, zorder=2))
    plate_art(ax)


_silhouette_cache = {}


def load_silhouette(hand, slot):
    # Drawn from the hitter's view, so the artwork for the opposite hand is used
    # (a RHP's arm appears on the left side of the picture).
    key = f"{'R' if hand == 'L' else 'L'}HP_{'High' if slot == 'High' else 'Low'}"
    if key not in _silhouette_cache:
        asset = SILHOUETTE_ASSETS[key]
        img = np.array(Image.open(io.BytesIO(base64.b64decode(asset["b64"]))).convert("RGBA"))
        _silhouette_cache[key] = (img, dict(asset["meta"]))
    return _silhouette_cache[key]


def draw_release_panel(fig, left, top, width, height, hand, slot, height_in, pitches, pitch_types):
    """Silhouette scaled to the pitcher's height, an 'arm' to his average release
    point, and the average release point of each pitch type."""
    ax = add_axes_in(fig, left, top, width, height)
    img, m = load_silhouette(hand, slot)

    scale = (height_in / SILHOUETTE_REF_HEIGHT) if SCALE_SILHOUETTE_BY_HEIGHT else 1.0
    head_in = SILHOUETTE_TOP_IN * scale
    inches_per_px = head_in / (m["feet"] - m["headtop"])
    k = inches_per_px
    # y = 0 at the feet, x = 0 at the middle of the body
    extent = [-m["cx"] * k, (m["w"] - m["cx"]) * k, (m["feet"] - m["h"]) * k, m["feet"] * k]
    ax.imshow(img, extent=extent, origin="upper", zorder=2, interpolation="bilinear")

    # Release point in inches. RelSide is flipped so it's plotted from the hitter's view.
    rel_side = to_number(pitches["RelSide"]) * 12
    rel_height = to_number(pitches["RelHeight"]) * 12
    ok = rel_side.notna() & rel_height.notna()
    avg_h, avg_s = rel_height[ok].mean(), rel_side[ok].mean()
    if ok.any():
        shoulder_x, shoulder_y = (m["ax"] - m["cx"]) * k, (m["feet"] - m["ay"]) * k
        ax.plot([shoulder_x, -avg_s], [shoulder_y, avg_h], color="black", lw=3.5,
                solid_capstyle="round", zorder=4)
        ax.scatter([-avg_s], [avg_h], s=55, facecolors="white", edgecolors="black", linewidths=1.6, zorder=5)
        for p in pitch_types:
            sel = pitches["TaggedPitchType"].eq(p) & ok
            if sel.any():
                ax.scatter(-rel_side[sel].mean(), rel_height[sel].mean(), s=34,
                           color=PITCH_COLORS.get(p, "gray"), edgecolors="black", linewidths=.7, zorder=6)

    ax.axhline(0, color="#555555", lw=1.2, zorder=1)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-4, 92)
    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xticks([])
    ax.tick_params(labelsize=6)
    ax.set_ylabel("Release height (in)", fontsize=7)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=.5, zorder=0)
    ax.set_title(f"Release Slot  •  {'LHP' if hand == 'L' else 'RHP'}  •  {slot} slot",
                 fontsize=10, fontweight="bold", pad=4)

    avg_text = "—" if not ok.any() else f"Avg release: {avg_h:.1f}″ H  |  {avg_s:.1f}″ S"
    return ax, avg_text


# ============================================================
# One page per pitcher
# ============================================================

def create_pitcher_page(pdf, pitcher, g, report_date, opponent):
    g = add_location_zones(g)
    info = CSUN_ROSTER.get(pitcher)
    if info is None:
        # Shouldn't happen (unknown pitchers are added to the roster at startup),
        # but fall back to the data's throwing hand and generic values.
        hand = normalize_hand(g["PitcherThrows"].iloc[0]) or "R"
        slot, height = "High", 72
    else:
        hand, slot, height = info["hand"], info["slot"], info["height"]

    # Keep tagged pitches that have at least some pitch information
    pitches = g[~g["TaggedPitchType"].isin(["Undefined", ""]) & g["TaggedPitchType"].notna()].copy()
    info_columns = ["PitchCall", "PlayResult", "KorBB", "TaggedPitchType", "RelSpeed",
                    "SpinRate", "InducedVertBreak", "HorzBreak", "PlateLocSide", "PlateLocHeight"]
    pitches = pitches[pitches[info_columns].notna().any(axis=1)]
    pitch_types = pitch_types_by_usage(pitches)
    n_types = max(len(pitch_types), 1)

    # ---- vertical layout ----
    HITTER_LABEL_TOP, HITTER_TOP = 0.95, 1.15
    ZONE_AX_TOP, ZONE_H = 2.35, 2.30
    ZONE_W = ZONE_H * 54 / 52                    # zone axes span x -27..27, y 5..57
    LOC_LABEL_TOP = ZONE_AX_TOP + ZONE_H + 0.42
    LOC_TOP = LOC_LABEL_TOP + 0.20
    CHAR_LABEL_TOP = 6.80
    CHAR_TOP = CHAR_LABEL_TOP + 0.20
    PAGE_BOTTOM = FIG_H - 0.20
    # Same row height as the Hitter Performance table, unless a pitcher throws so
    # many pitch types that the tables would run off the page.
    row_h = min(CELL_H,
                (CHAR_LABEL_TOP - 0.10 - LOC_TOP) / (n_types + 2),
                (PAGE_BOTTOM - CHAR_TOP) / (n_types + 1))

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    # ---- header ----
    ax_logo = add_axes_in(fig, MARGIN_L, LOGO_TOP_IN, LOGO_WIDTH_IN, LOGO_WIDTH_IN * LOGO_ASPECT)
    ax_logo.axis("off")
    ax_logo.imshow(np.array(Image.open(io.BytesIO(base64.b64decode(LOGO_B64))).convert("RGBA")),
                   aspect="equal", interpolation="bilinear")
    fig.text(.5, 1 - 0.30 / FIG_H, f"{pitcher}  |  PITCHING REPORT", ha="center", va="center",
             fontsize=16, fontweight="bold")
    fig.text(.5, 1 - 0.60 / FIG_H, f"{report_date}   •   {opponent}   •   {len(g)} pitches",
             ha="center", va="center", fontsize=9, color="#444444")

    # ---- hitter performance ----
    performance = []
    for label, sub in [("Overall", pitches),
                       ("vs RHH", pitches[pitches["BatterSide"].eq("Right")]),
                       ("vs LHH", pitches[pitches["BatterSide"].eq("Left")])]:
        performance.append([label] + list(hitter_summary(sub).values()))
    performance = pd.DataFrame(performance, columns=["Split", "BAA", "AvgEV", "GB%", "FB+PU%",
                                                     "Swing%", "Whiff%", "Chase%"])
    label_in(fig, MARGIN_L, HITTER_LABEL_TOP, "Hitter Performance")
    draw_table(fig, performance, MARGIN_L, HITTER_TOP, LEFT_COL_W, CELL_H, fontsize=8)

    # ---- pitch locations vs RHH / LHH (pitcher's view) ----
    for i, (side_label, side_value) in enumerate([("vs RHH", "Right"), ("vs LHH", "Left")]):
        sub = pitches[pitches["BatterSide"].eq(side_value)]
        shown, n_located = sample_zone_pitches(sub)
        center = MARGIN_L + LEFT_COL_W * (0.25 + 0.5 * i) + 0.12
        ax = add_axes_in(fig, center - ZONE_W / 2, ZONE_AX_TOP, ZONE_W, ZONE_H)
        ax.set_title(f"Pitch Location {side_label}", fontsize=9, fontweight="bold", pad=4)
        for p in pitch_types:
            x = shown[shown["TaggedPitchType"].eq(p)]
            if len(x):
                ax.scatter(x["LocSideIn"], x["LocHeightIn"], s=22, color=PITCH_COLORS.get(p, "gray"),
                           alpha=.8, edgecolors="none", zorder=4)
        draw_zone(ax)
        ax.set_xlim(-27, 27)
        ax.set_ylim(5, 57)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Horizontal location (in) — pitcher's view", fontsize=6.5)
        ax.set_ylabel("Height (in)", fontsize=7)
        ax.tick_params(labelsize=6)
        if n_located > len(shown):
            ax.text(-26, 56.3, f"{len(shown)} of {n_located} pitches shown (usage-weighted)",
                    fontsize=5.5, color="#666666", va="top", ha="left")

    # ---- location breakdown table ----
    location = pitch_location_table(pitches, pitch_types)
    loc_w = CELL_W * len(location.columns)
    loc_left = MARGIN_L + (LEFT_COL_W - loc_w) / 2
    label_in(fig, loc_left, LOC_LABEL_TOP, "Pitch Location Breakdown")
    draw_table(fig, location, loc_left, LOC_TOP, loc_w, row_h, fontsize=8, header_fontsize=7.5)

    # ---- release slot (right column, top) ----
    MV_S = 2.20
    mv_left = FIG_W - 0.30 - MV_S
    REL_TOP = HITTER_TOP
    REL_H = 2.30
    REL_W = REL_H * 110 / 96                    # axes span x -55..55, y -4..92
    rel_left = mv_left + MV_S / 2 - REL_W / 2 - 0.12
    _, avg_text = draw_release_panel(fig, rel_left, REL_TOP, REL_W, REL_H, hand, slot, height,
                                     pitches, pitch_types)
    mid = rel_left + REL_W / 2
    fig.text(mid / FIG_W, 1 - (REL_TOP + REL_H + 0.06) / FIG_H, avg_text, ha="center", va="top", fontsize=7)
    fig.text(mid / FIG_W, 1 - (REL_TOP + REL_H + 0.22) / FIG_H,
             f"Player height: {int(height) // 12}′{int(height) % 12}″", ha="center", va="top", fontsize=7)

    # ---- movement plot (right column, bottom) ----
    MV_TOP = REL_TOP + REL_H + 0.80
    ax = add_axes_in(fig, mv_left, MV_TOP, MV_S, MV_S)
    ax.set_title("Pitch Movement", fontsize=10, fontweight="bold", pad=4)
    for p in pitch_types:
        x = pitches[pitches["TaggedPitchType"].eq(p)]
        hb, ivb = to_number(x["HorzBreak"]), to_number(x["InducedVertBreak"])
        valid = hb.notna() & ivb.notna()
        if valid.any():
            ax.scatter(hb[valid], ivb[valid], s=22, alpha=.6, color=PITCH_COLORS.get(p, "gray"),
                       edgecolors="none", zorder=3)
        d1 = D1_MOVEMENT_AVERAGES.get(hand, {}).get(p)
        if d1:
            ax.scatter(d1["HB"], d1["IVB"], s=150, facecolors="none", edgecolors=PITCH_COLORS.get(p, "gray"),
                       linewidths=2.0, marker="o", zorder=5)
    ax.axhline(0, color="gray", lw=.7, ls="--")
    ax.axvline(0, color="gray", lw=.7, ls="--")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal", adjustable="box")
    ticks = np.arange(-25, 26, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(color="#E6E6E6", lw=.5, zorder=0)
    ax.set_xlabel("Horizontal Break (in)", fontsize=8)
    ax.set_ylabel("Induced Vertical Break (in)", fontsize=8)
    ax.tick_params(labelsize=6)

    # ---- legend (between the zones and the movement plot) ----
    def has_plot_data(p):
        x = pitches[pitches["TaggedPitchType"].eq(p)]
        has_movement = (to_number(x["HorzBreak"]).notna() & to_number(x["InducedVertBreak"]).notna()).any()
        has_location = (to_number(x["PlateLocSide"]).notna() & to_number(x["PlateLocHeight"]).notna()).any()
        return bool(has_movement or has_location)

    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PITCH_COLORS.get(p, "gray"),
                      markeredgecolor="none", markersize=7, label=p)
               for p in pitch_types if has_plot_data(p)]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                          markeredgecolor="black", markeredgewidth=1.5, markersize=9, label="D1 avg"))
    ax_legend = add_axes_in(fig, MARGIN_L + LEFT_COL_W + 0.08, MV_TOP, 0.95, 1.9)
    ax_legend.axis("off")
    ax_legend.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1), frameon=False,
                     fontsize=7.5, title="Pitch Type", title_fontsize=8, labelspacing=.7,
                     handletextpad=.3, borderaxespad=0)

    # ---- pitch characteristics (full width) ----
    characteristics = pitch_characteristics(pitches, pitch_types)
    label_in(fig, MARGIN_L, CHAR_LABEL_TOP, "Pitch Characteristics")
    draw_table(fig, characteristics, MARGIN_L, CHAR_TOP, FULL_W, row_h, fontsize=8, header_fontsize=7.5,
               red_headers=("Stuff+",))

    pdf.savefig(fig)
    plt.close(fig)


# ============================================================
# Run
# ============================================================

def ask_about_unknown_pitchers(df):
    """For pitchers not in CSUN_ROSTER, ask for arm slot and height (hand comes
    from the data) and add them to the roster for this run."""
    unknown = sorted(set(df["Pitcher"].dropna().astype(str)) - set(CSUN_ROSTER))
    if not unknown:
        return

    answers = {}
    print("Pitcher(s) not found in CSUN_ROSTER. Answer two quick questions for each (type, then press Enter).")
    for name in unknown:
        close = difflib.get_close_matches(name, list(CSUN_ROSTER), n=1, cutoff=0.7)
        hint = (f"  (closest roster name: {close[0]!r} — if that's the same person, the CSV spells "
                f"the name differently than CSUN_ROSTER)") if close else ""
        print(f"\n{name!r}" + hint)

        hand = normalize_hand(df.loc[df["Pitcher"].eq(name), "PitcherThrows"].dropna().iloc[0]) or "R"
        print(f"  Throw hand from the data: {hand}HP")

        slot = ""
        while slot not in ("High", "Low"):
            reply = input("  Arm slot — type H for High or L for Low: ").strip().lower()
            slot = {"h": "High", "high": "High", "l": "Low", "low": "Low"}.get(reply, "")

        height = 72
        reply = input("  Height in inches (press Enter for 72): ").strip()
        if reply.isdigit():
            height = int(reply)
        answers[name] = {"hand": hand, "slot": slot, "height": height}

    print("\nTo save these permanently, add to CSUN_ROSTER:")
    for name, a in answers.items():
        print(f' {name!r}:{{"hand":"{a["hand"]}","slot":"{a["slot"]}","height":{a["height"]}}},')
    print()
    CSUN_ROSTER.update(answers)


def main():
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No CSV uploaded.")
    input_name = next(iter(uploaded))
    with open(input_name, "wb") as f:
        f.write(uploaded[input_name])

    df, metadata = read_trackman_csv(input_name)
    required = {"PitcherTeam", "Pitcher", "TaggedPitchType", "PitchCall", "PlateLocSide",
                "PlateLocHeight", "PitcherThrows", "BatterSide"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[df["PitcherTeam"].astype(str).str.strip().eq(TEAM_CODE)].copy()
    if df.empty:
        raise ValueError(f"No rows found for PitcherTeam == {TEAM_CODE}. Check the team code.")

    report_date, opponent, opponents, dates, n_games = summarize_games(df, metadata)
    if len(dates) > 1 or len(opponents) > 1 or n_games > 1:
        print(f"Cumulative file detected: {n_games} game(s), {report_date}")
        print(f"Opposing teams ({len(opponents)}): {', '.join(opponents)}")
        print(f"Game dates ({len(dates)}): {', '.join(pd.Timestamp(d).strftime('%m-%d-%Y') for d in dates)}")
        opponent = f"{n_games} games • {opponent}"

    # Match names to the roster even if the CSV has stray spaces or different capitalization
    roster_by_key = {name_key(k): k for k in CSUN_ROSTER}
    df["Pitcher"] = df["Pitcher"].map(
        lambda s: roster_by_key.get(name_key(s), re.sub(r"\s+", " ", str(s)).strip()) if pd.notna(s) else s)

    ask_about_unknown_pitchers(df)

    pdf_name = "CSUN_Pitcher_Reports_" + re.sub(r"[^0-9A-Za-z]+", "_", report_date).strip("_") + ".pdf"
    with PdfPages(pdf_name) as pdf:
        for pitcher in sorted(df["Pitcher"].dropna().unique()):
            create_pitcher_page(pdf, pitcher, df[df["Pitcher"].eq(pitcher)].copy(), report_date, opponent)

    print(f"Created {pdf_name}")
    files.download(pdf_name)


# ============================================================
# Embedded images
# ============================================================
# CSUN logo and the four arm-slot silhouettes (PNG, base64). "meta" gives each
# silhouette's size, the head and feet rows, the body's center line (cx) and the
# shoulder point (ax, ay) where the arm is attached, all in image pixels.

LOGO_B64 = (
    "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"
    "6/8Aze9Ey4+6r2sAAAAASUVORK5CYII="
)

SILHOUETTE_ASSETS = {
    "LHP_Low": {
        "meta": {"w": 144, "h": 352, "headtop": 0.0, "feet": 351.1, "cx": 43.9, "ax": 26.0, "ay": 66.4},
        "b64": (
            "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"
        )},
    "RHP_Low": {
        "meta": {"w": 144, "h": 352, "headtop": 0.0, "feet": 351.1, "cx": 99.6, "ax": 116.4, "ay": 66.5},
        "b64": (
            "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"
            "SUVORK5CYII="
        )},
    "LHP_High": {
        "meta": {"w": 182, "h": 369, "headtop": 0.0, "feet": 368.1, "cx": 53.5, "ax": 79.1, "ay": 40.6},
        "b64": (
            "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"
        )},
    "RHP_High": {
        "meta": {"w": 182, "h": 369, "headtop": 0.0, "feet": 368.1, "cx": 127.1, "ax": 100.0, "ay": 41.1},
        "b64": (
            "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"
        )},
}


if __name__ == "__main__":
    main()
