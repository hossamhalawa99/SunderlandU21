# app.py
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict
from io import BytesIO
import base64
from pathlib import Path
from datetime import datetime, date
from supabase import create_client, Client

# --- PDF: reportlab ---
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# --- Charts: matplotlib ---
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re

# =============== CONFIG ===============
TARGET_TABLE = "sunderlandu21_stats"

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "sunderland_U21_logo.jpeg"

# Supabase (embedded creds; for production, move to secrets)
SUPABASE_URL = "https://cnrenslmevvzjvrenbrp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNucmVuc2xtZXZ2emp2cmVuYnJwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIyODExMzEsImV4cCI6MjA3Nzg1NzEzMX0.cytLc4N6Ym8k-crercH5jiiQ5svEUM47DoD9QBEbfHg"


CLUB_NAME = "Sunderland U21"  # used to derive opponent
DATE_COL = "date"             # matches your ODS

# =============== INIT ===============
st.set_page_config(
    page_title="Sunderland U21 Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============== STYLE ===============
st.markdown("""
<style>
:root{
  --bg:#0a0f1a; --panel:#0f172a; --panel-2:#0b1220; --muted:#9fb2d9;
  --text:#e5e7eb; --accent:#DC2626; --accent-2:#B91C1C;
  --line:#23324d;
}
html, body{
  font-family: "Segoe UI", Tahoma, sans-serif !important;
  color:var(--text) !important; background:var(--bg) !important;
}
[data-testid="stSidebar"]{
  width:340px !important; min-width:340px !important;
  background: linear-gradient(180deg, #0b1220, #0a1020);
  border-inline-end: 1px solid var(--line);
}
.hr-accent{ height:2px; border:0; background:linear-gradient(90deg, transparent, var(--accent), transparent); margin: 8px 0 14px; }
.fin-head{
  display:flex; justify-content:space-between; align-items:center;
  border: 1px dashed rgba(220,38,38,.35); border-radius:16px;
  padding: 16px 18px; margin:8px 0 14px; background:linear-gradient(180deg,#0b1220,#0e1424);
}
.fin-head .line{ font-size:22px; font-weight:900; color:var(--text); }
.badge{ background:var(--accent); color:#fff; padding:6px 12px; border-radius:999px; font-weight:700; }
[data-testid="stDataFrame"] thead tr th{
  position: sticky; top: 0; z-index: 2;
  background: #132036; color: #e7eefc; font-weight:800; font-size:15px;
  border-bottom: 1px solid var(--line);
}
.hsec{ color:#e7eefc; font-weight:900; margin:6px 0 10px; font-size: 22px; }
</style>
""", unsafe_allow_html=True)

# =============== UTILS ===============
def _logo_html() -> str:
    if not LOGO_PATH.exists():
        return ""
    try:
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        return f'<img src="data:image/jpeg;base64,{b64}" width="64" />'
    except Exception:
        return ""

@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        return None

def _safe_date(x) -> Optional[date]:
    try:
        if isinstance(x, str):
            return datetime.fromisoformat(x).date()
        if isinstance(x, (datetime, date)):
            return x if isinstance(x, date) else x.date()
    except Exception:
        return None
    return None

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    d = df.copy()

    # normalize column names expected by this app
    # ODS has: 'scheme', 'date', 'home_team','away_team','result_of_match','goals','xg','ppda',...
    # create opponent relative to CLUB_NAME
    d["is_home"] = (d.get("home_team", "") == CLUB_NAME)
    d["opponent"] = np.where(d["is_home"], d.get("away_team", ""), d.get("home_team", ""))

    # Coerce numeric where sensible
    numeric_cols = [
        "goals","xg","ppda","shots","shots_on_target","won_duels_percentage",
        "won_defensive_duels_percentage","won_aerial_duels_percentage",
        "touches_in_penalty_area","total_shots_against","percentage_of_possession",
        "positional_attacks","counterattacks"
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Parse dates
    if DATE_COL in d.columns:
        d[DATE_COL] = d[DATE_COL].apply(_safe_date)

    # Result marker for timeline
    res_map = {"win":"W","Win":"W","W":"W","draw":"D","Draw":"D","D":"D","loss":"L","Loss":"L","L":"L"}
    d["result_mark"] = d.get("result_of_match","").map(lambda x: res_map.get(str(x), str(x)))

    # Match label fallback
    if "match" not in d.columns:
        d["match"] = d.apply(lambda r: f'{r.get("home_team","")} vs {r.get("away_team","")}', axis=1)

    return d

# =============== DATA ===============
@st.cache_data(ttl=600)
def fetch_match_data(
    selected_comp: Optional[List[str]] = None,
    selected_opponents: Optional[List[str]] = None,
    selected_scheme: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    supabase = get_supabase_client()
    if supabase is None:
        return pd.DataFrame()

    try:
        q = supabase.table(TARGET_TABLE).select("*")

        # server-side filters that match ODS columns
        if selected_comp and "All" not in selected_comp:
            q = q.in_("competition", selected_comp)
        if selected_scheme and "All" not in selected_scheme:
            q = q.in_("scheme", selected_scheme)
        if date_from:
            q = q.gte(DATE_COL, date_from)
        if date_to:
            q = q.lte(DATE_COL, date_to)
        q = q.order(DATE_COL, desc=True)

        res = q.execute()
        data = res.data or []
        df = pd.DataFrame(data)
        df = add_derived_columns(df)

        # client-side filter for opponents (derived)
        if selected_opponents and "All" not in selected_opponents and "opponent" in df.columns:
            df = df[df["opponent"].isin(selected_opponents)]

        return df
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_distinct(column: str) -> List[str]:
    supabase = get_supabase_client()
    if supabase is None:
        return []
    try:
        res = supabase.table(TARGET_TABLE).select(column).execute()
        df = pd.DataFrame(res.data or [])
        if column in df.columns:
            vals = sorted({str(v) for v in df[column].dropna().tolist()})
            return vals
        return []
    except Exception:
        return []

def collect_opponent_options() -> List[str]:
    """Build opponent list from home/away columns relative to CLUB_NAME."""
    supabase = get_supabase_client()
    if supabase is None:
        return []
    try:
        res = supabase.table(TARGET_TABLE).select("home_team,away_team").execute()
        df = pd.DataFrame(res.data or [])
        if df.empty:
            return []
        df = add_derived_columns(df)
        vals = sorted({str(v) for v in df["opponent"].dropna().tolist()})
        return vals
    except Exception:
        return []

# =============== CHARTS (PNG in-memory) ===============
def _fig_bytes(fig) -> bytes:
    bio = BytesIO()
    fig.tight_layout()
    fig.savefig(bio, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()

def _x_labels_from_opponent(df: pd.DataFrame) -> List[str]:
    """
    Return cleaned full match labels:
      - prefer 'match' column (Home Team vs Away Team)
      - fallback to 'opponent' if missing
      - replace 'vs' (with optional dots/spaces) with ':'
      - remove 'U21' (case-insensitive)
      - remove all whitespace
    """
    if "match" in df.columns and not df["match"].dropna().empty:
        raw = df["match"].astype(str).tolist()
    elif "opponent" in df.columns and not df["opponent"].dropna().empty:
        raw = df["opponent"].astype(str).tolist()
    else:
        return [str(i) for i in range(len(df))]

    cleaned = []
    for s in raw:
        s = re.sub(r'(?i)\s*vs\.?\s*', ':', s)   # replace "vs" variants with ":"
        s = re.sub(r'(?i)u21', '', s)            # remove U21 (case-insensitive)
        s = re.sub(r'\s+', '', s)                # remove all whitespace
        cleaned.append(s)
    return cleaned

def _short_labels(labels: List[str], max_len: int = 25) -> List[str]:
    """Truncate long labels for x-axis display."""
    out: List[str] = []
    for s in labels:
        if s is None:
            out.append("")
            continue
        s = str(s)
        if len(s) <= max_len:
            out.append(s)
        else:
            out.append(s[: max_len - 3] + "...")
    return out

def chart_results_timeline(df: pd.DataFrame) -> bytes:
    d = df.sort_values(DATE_COL).copy()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    # scatter by result
    colors_map = {"W":"#22c55e","D":"#f59e0b","L":"#ef4444"}
    x = d[DATE_COL]
    y = np.zeros(len(d))
    for r in ["W","D","L"]:
        m = d["result_mark"] == r
        ax.scatter(x[m], y[m], s=120, alpha=.9, label=r, c=colors_map.get(r, "#60a5fa"), edgecolors="#0f172a")
    ax.set_yticks([])
    ax.set_title("Match Results Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=.3)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    return _fig_bytes(fig)

def chart_goals_vs_xg(df: pd.DataFrame) -> bytes:
    # X = opponent / match name
    if df.empty:
        return b""
    d = df.copy()
    labels = _x_labels_from_opponent(d)
    x = np.arange(len(labels))

    # Team goals and xG (for)
    goals = pd.to_numeric(d.get("goals", pd.Series([0]*len(d))), errors="coerce").fillna(0).to_numpy(dtype=float)
    xg = None
    for col in ("xg","xg_for","xG_for","xG"):
        if col in d.columns:
            xg = pd.to_numeric(d[col], errors="coerce").fillna(0).to_numpy(dtype=float)
            break
    if xg is None:
        xg = np.zeros(len(d), dtype=float)

    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width/2, goals, width=width, label="Team", color="#ef4444", alpha=0.9)
    ax.bar(x + width/2, xg, width=width, label="xG (Team)", color="#2563eb", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(labels, max_len=25), rotation=45, ha="right", fontsize=9)
    ax.set_title("Team Goals vs Expected Goals (xG)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _fig_bytes(fig)

def chart_ppda_trend(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    d = df.copy()
    labels = _x_labels_from_opponent(d)
    x = np.arange(len(labels))
    if "ppda" not in d.columns:
        return b""
    y = pd.to_numeric(d["ppda"], errors="coerce").fillna(np.nan).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(x, y, marker="o", color="#7c3aed")
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(labels, max_len=25), rotation=45, ha="right", fontsize=9)
    ax.set_title("PPDA Trend (Team)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _fig_bytes(fig)

def chart_duel_win(df: pd.DataFrame) -> bytes:
    # choose first available duel-win column and plot using opponent names on x axis
    if df.empty:
        return b""
    d = df.copy()
    labels = _x_labels_from_opponent(d)
    x = np.arange(len(labels))
    col = None
    for c in ["won_duels_percentage","won_defensive_duels_percentage","won_aerial_duels_percentage"]:
        if c in d.columns:
            col = c; break
    if not col:
        return b""
    y = pd.to_numeric(d[col], errors="coerce").fillna(0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(x, y, marker="o", color="#16a34a")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(labels, max_len=25), rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Duel Win % (Team) — {col.replace('_',' ').title()}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _fig_bytes(fig)

def chart_final_third(df: pd.DataFrame) -> bytes:
    cols = [c for c in ["touches_in_penalty_area","positional_attacks","counterattacks"] if c in df.columns]
    if not cols or df.empty:
        return b""
    d = df.copy()
    labels = _x_labels_from_opponent(d)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4))
    for c in cols:
        y = pd.to_numeric(d[c], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label=c.replace("_"," ").title())
    ax.set_title("Final Third Activity (Team)")
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(labels, max_len=25), rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _fig_bytes(fig)

def chart_defense_shots_against(df: pd.DataFrame) -> bytes:
    col = "total_shots_against"
    if col not in df.columns or df.empty:
        return b""
    d = df.copy()
    labels = _x_labels_from_opponent(d)
    x = np.arange(len(labels))
    y = pd.to_numeric(d[col], errors="coerce").fillna(0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(x, y, color="#f97316")
    ax.set_title("Shots Against (Team)")
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(labels, max_len=25), rotation=45, ha="right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _fig_bytes(fig)

def chart_match_overview(df: pd.DataFrame) -> bytes:
    """
    Compact overview:
      - xG (team) line by match (if xG column exists)
      - Possession % by match (if possession column exists)
    Uses cleaned full match label on x-axis (vs -> ':' , remove U21 and whitespace).
    Panels show only if their data exists.
    """
    if df.empty:
        return b""
    d = df.copy()

    # Labels (cleaned) and x positions
    labels = _x_labels_from_opponent(d)
    x = np.arange(len(labels))

    # Team xG (only team columns, fallback: not present)
    xg_cols = [c for c in ("xg","xg_for","xG_for","xG") if c in d.columns]
    has_xg = len(xg_cols) > 0
    if has_xg:
        xg_arr = pd.to_numeric(d[xg_cols[0]], errors="coerce").fillna(0).to_numpy(dtype=float)

    # Possession (team)
    poss_cols = [c for c in ("percentage_of_possession","possession","possession_pct") if c in d.columns]
    has_poss = len(poss_cols) > 0
    if has_poss:
        poss_arr = pd.to_numeric(d[poss_cols[0]], errors="coerce").fillna(0).to_numpy(dtype=float)

    # If neither available, skip
    if not has_xg and not has_poss:
        return b""

    # Colors by result if available (used for possession bars)
    color_map = {"W": "#22c55e", "D": "#f59e0b", "L": "#ef4444"}
    if "result_mark" in d.columns:
        colors = d["result_mark"].map(lambda x: color_map.get(str(x), "#60a5fa")).tolist()
    else:
        colors = ["#60a5fa"] * len(d)

    # Build dynamic layout: 1 or 2 panels depending on available data
    n_panels = (1 if has_xg else 0) + (1 if has_poss else 0)
    height = 3.5 * n_panels if n_panels > 0 else 3.5
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, max(4, height)), sharex=True)
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if has_xg:
        ax = axes[idx]
        ax.plot(x, xg_arr, marker="o", color="#ef4444", linewidth=2, label="Team")
        ax.set_ylabel("xG")
        ax.set_title("Expected Goals (xG) by Match")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left")
        idx += 1

    if has_poss:
        ax = axes[idx]
        ax.bar(x, poss_arr, color=colors)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Possession %")
        ax.set_title("Possession Percentage by Match")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # X-axis labels on final axis
        MAX_LABEL = 30
        short_labels = [lab if len(lab) <= MAX_LABEL else lab[:MAX_LABEL-3] + "..." for lab in labels]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    return _fig_bytes(fig)


def chart_radar_for_match(single_row_df: pd.DataFrame, reference_df: pd.DataFrame) -> bytes:
    """
    Radar for a single match (single_row_df must be one-row DataFrame).
    Uses column aliases and only includes metrics that exist both in the single row
    and the reference dataframe. Scales each metric to the max value in reference_df.
    Returns PNG bytes or b'' if nothing to plot.
    """
    if single_row_df is None or single_row_df.empty:
        return b""
    row = single_row_df.iloc[0:1]

    # candidate metric groups (aliases) -> label
    metrics_candidates = [
        (["percentage_of_possession", "possession", "possession_pct"], "Possession"),
        (["xg", "xg_for", "xG_for", "xG"], "xG"),
        (["ppda", "ppda_allow", "ppda_allowed"], "PPDA"),
        (["shots", "total_shots"], "Shots"),
        (["shots_on_target", "shots_on_target_total"], "Shots on Target"),
        (["touches_in_penalty_area", "touches_in_box", "tp_in_penalty_area"], "Touches in Pen Area"),
        (["won_duels_percentage", "duel_win_pct", "won_duels"], "Duel Win %"),
    ]

    labels = []
    values = []
    max_vals = []

    ref_cols = set(reference_df.columns.astype(str)) if reference_df is not None else set()
    row_cols = set(row.columns.astype(str))

    for aliases, label in metrics_candidates:
        found = None
        for a in aliases:
            if a in ref_cols and a in row_cols:
                found = a
                break
        if not found:
            # also accept if present in row only (use row value and avoid scale issues)
            for a in aliases:
                if a in row_cols:
                    found = a
                    break
        if not found:
            continue

        # get numeric values
        ref_max = None
        if reference_df is not None and found in reference_df.columns:
            ref_max = pd.to_numeric(reference_df[found], errors="coerce").max(skipna=True)
        if pd.isna(ref_max) or ref_max == 0 or ref_max is None:
            ref_max = pd.to_numeric(row[found], errors="coerce").fillna(0).max()
            if pd.isna(ref_max) or ref_max == 0:
                ref_max = 1.0
        val = float(pd.to_numeric(row[found], errors="coerce").fillna(0).iloc[0])
        labels.append(label)
        values.append(val)
        max_vals.append(float(ref_max))

    if not labels:
        return b""

    # Scale 0-100
    scaled = [(v / m) * 100.0 if m and m != 0 else 0.0 for v, m in zip(values, max_vals)]

    # Close polygon
    angles = np.linspace(0, 2 * np.pi, len(scaled), endpoint=False).tolist()
    scaled += [scaled[0]]
    angles += [angles[0]]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, scaled, color="#ef4444", linewidth=2)
    ax.fill(angles, scaled, color="#ef4444", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 100)

    # Safe extraction of match label using positional access
    match_label = ""
    try:
        if "match" in row.columns:
            match_label = str(row["match"].iat[0])
    except Exception:
        match_label = ""

    ax.set_title(f"Performance Radar — {match_label}", y=1.08)
    ax.grid(True, linestyle="--", alpha=0.3)
    return _fig_bytes(fig)

def chart_placeholder_radar(match_name: str) -> bytes:
    """Return a small placeholder PNG indicating no metrics for this match."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.6, "No numeric metrics available", ha="center", va="center", fontsize=12, color="#7b8794")
    ax.text(0.5, 0.45, match_name, ha="center", va="center", fontsize=10, color="#222222")
    ax.set_title("Performance Radar — No Data", fontsize=11)
    return _fig_bytes(fig)

def chart_rolling_averages(df: pd.DataFrame) -> bytes:
    """
    Multi-panel rolling averages chart (omits xGA).
    Uses rolling window from session_state['rolling_window'] (default 3).
    """
    if df.empty:
        return b""
    d = df.sort_values(DATE_COL).copy()

    try:
        window = int(st.session_state.get("rolling_window", 3))
    except Exception:
        window = 3
    window = max(1, min(window, 20))

    metric_candidates = [
        (["xg", "xg_for", "xG_for", "xG"], "Expected Goals (xG)"),
        (["percentage_of_possession", "possession", "possession_pct"], "Possession %"),
        (["pass_accuracy", "pass_accuracy_pct", "pass_accuracy_percent"], "Pass Accuracy %"),
        (["ppda", "ppda_allow", "ppda_allowed", "ppda"], "PPDA"),
        (["won_duels_percentage", "duel_win_pct", "won_duels"], "Duel Win %"),
    ]

    found_metrics = []
    for aliases, label in metric_candidates:
        found = None
        for a in aliases:
            if a in d.columns:
                found = a
                break
        if found:
            ser = pd.to_numeric(d[found], errors="coerce")
            found_metrics.append((found, label, ser))

    if not found_metrics:
        return b""

    labels = _x_labels_from_opponent(d)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()
    colors = ["#ef4444", "#7c3aed", "#10b981", "#f59e0b", "#2563eb"]

    for i, (col, label, ser) in enumerate(found_metrics):
        ax = axes_flat[i]
        raw = ser.fillna(np.nan)
        roll = raw.rolling(window=window, min_periods=1).mean()

        ax.plot(x, roll.to_numpy(), marker="o", color=colors[i % len(colors)], linewidth=2, label=f"{window}-Match Avg")
        ax.scatter(x, raw.to_numpy(), color=colors[i % len(colors)], alpha=0.25, s=20)

        overall = raw.mean(skipna=True)
        if not np.isnan(overall):
            ax.axhline(overall, color="gray", linestyle="--", alpha=0.6, linewidth=1)
            ax.text(0.98, 0.9, f"Overall Avg: {overall:.2f}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=9, color="gray", bbox=dict(facecolor="white", alpha=0.0))

        ax.set_ylabel("Value")
        ax.set_title(label)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(_short_labels(labels, max_len=20), rotation=45, ha="right", fontsize=8)
        if "poss" in label.lower() or "pass" in label.lower() or "%" in label:
            ax.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=8)

    for j in range(len(found_metrics), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"SAFC U21 Rolling Averages (Window={window})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _fig_bytes(fig)


# Ensure VISUALS registry exists and includes all charts
from typing import Callable
VISUALS: Dict[str, Callable[[pd.DataFrame], bytes]] = {
    "Match Overview (xG · Possession)": chart_match_overview,
    "Goals vs xG (Team)": chart_goals_vs_xg,
    "Rolling Averages (by window)": chart_rolling_averages,
    "PPDA Trend (Team)": chart_ppda_trend,
    "Duel Win % (Team)": chart_duel_win,
    "Final Third Activity (Team)": chart_final_third,
    "Shots Against (Team)": chart_defense_shots_against,
}



# =============== PDF BUILDER ===============
def create_pdf_report(
    df: pd.DataFrame,
    title: str,
    logo_path: Path,
    radar_images: List[bytes],
    chart_images: List[bytes],
    include_table: bool = True
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=title
    )
    styles = getSampleStyleSheet()
    elements = []

    # Header: larger logo + title + generated-by line
    if logo_path.exists():
        try:
            logo = RLImage(str(logo_path))
            # make logo larger for PDF
            logo.drawHeight = 0.9 * inch
            logo.drawWidth = logo.imageWidth * (logo.drawHeight / logo.imageHeight)
            logo.hAlign = 'LEFT'
            elements.append(logo)
        except Exception:
            pass

    title_style = ParagraphStyle(
        name="TitleStyle",
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,
        spaceAfter=0.06 * inch
    )
    subtitle = ParagraphStyle(
        name="Sub",
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=1
    )
    author_style = ParagraphStyle(
        name="Author",
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1,
        spaceAfter=0.12 * inch
    )

    elements.append(Paragraph(title, title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle))
    elements.append(Paragraph("Generated by: Hossam Halawa", author_style))
    elements.append(Spacer(1, 0.12 * inch))

    # First: put all radars on a single page (if any) arranged 3-per-row
    if radar_images:
        elements.append(Paragraph("Performance Radars", styles['Heading2']))
        elements.append(Spacer(1, 0.06 * inch))
        # Build RLImage objects sized to fit 3 per row
        rl_imgs = []
        for img in radar_images:
            try:
                rl = RLImage(BytesIO(img), width=3.2 * inch, height=3.2 * inch)
                rl.hAlign = 'CENTER'
                rl_imgs.append(rl)
            except Exception:
                continue
        # arrange into rows of 3 and pad last row
        rows = []
        for i in range(0, len(rl_imgs), 3):
            row = rl_imgs[i:i+3]
            while len(row) < 3:
                row.append('')  # empty cell
            rows.append(row)
        if rows:
            tbl = Table(rows, colWidths=[3.2 * inch] * 3, hAlign='CENTER')
            tbl.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            elements.append(tbl)
        elements.append(PageBreak())

    # Then: other charts, two per page (side-by-side stacking)
    for i, img in enumerate(chart_images):
        try:
            elements.append(RLImage(BytesIO(img), width=9.5*inch, height=3.6*inch))
            elements.append(Spacer(1, 0.12 * inch))
            # page break every 2 charts to avoid cramming
            if (i + 1) % 2 == 0:
                elements.append(PageBreak())
        except Exception:
            continue

    # Table (optional) - keep at end
    if include_table and not df.empty:
        elements.append(PageBreak())
        elements.append(Paragraph("Filtered Match Data", styles['Heading2']))
        cols = [c for c in ["date","competition","match","result_of_match","home_team","away_team","score","goals","xg","ppda"] if c in df.columns]
        slim = df[cols].copy()
        slim = slim.fillna("")
        slim = slim.head(60)
        data_tbl = [slim.columns.tolist()] + slim.astype(str).values.tolist()

        table = Table(data_tbl, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DC2626')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#333333')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ]))
        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# =============== HEADER ===============
c_logo, c_title = st.columns([1, 6], gap="small")
with c_logo:
    st.markdown(_logo_html(), unsafe_allow_html=True)
with c_title:
    st.markdown("""
<h1 style="color:#e7eefc; font-weight:900; margin:0;">
  Sunderland U21 Customized Dashboard
  <span style="font-size:18px; color:#9fb2d9; font-weight:600;">| Match Stats</span>
</h1>
""", unsafe_allow_html=True)
st.markdown('<hr class="hr-accent"/>', unsafe_allow_html=True)

# =============== SIDEBAR FILTERS ===============
with st.sidebar:
    st.markdown('<h2 style="color:#e7eefc; font-weight:900;">Filter Data</h2>', unsafe_allow_html=True)
    st.markdown('<hr class="hr-accent"/>', unsafe_allow_html=True)

    comp_options = ["All"] + get_distinct("competition")
    selected_comp = st.multiselect("Competition", options=comp_options, default=["All"], key="comp_filter")

    # Location filter
    st.markdown('<h2 style="color:#e7eefc; font-weight:900; margin-top:12px;">Match Location</h2>', unsafe_allow_html=True)
    location_option = st.selectbox("Location", options=["All", "Home", "Away"], index=0, help="Filter matches by Home/Away relative to the club")

    # Date range
    st.markdown('<h2 style="color:#e7eefc; font-weight:900; margin-top:16px;">Date Range</h2>', unsafe_allow_html=True)
    apply_date = st.checkbox("Apply date filter", value=False)
    if apply_date:
        col_from, col_to = st.columns(2)
        with col_from:
            d_from = st.date_input("From", value=date(2020,1,1))
        with col_to:
            d_to = st.date_input("To", value=datetime.now().date())
        date_from = d_from.strftime("%Y-%m-%d")
        date_to = d_to.strftime("%Y-%m-%d")
    else:
        date_from = None
        date_to = None

    # Rolling average window control (user-selected)
    st.markdown('<h2 style="color:#e7eefc; font-weight:900; margin-top:12px;">Rolling Window</h2>', unsafe_allow_html=True)
    rolling_window = st.slider("Rolling average window (matches)", min_value=1, max_value=10, value=3, step=1, key="rolling_window", help="Number of matches to use for rolling averages")

    st.markdown("---")

    # Fetch data now (so we can populate match selector immediately)
    df_sidebar = fetch_match_data(
        selected_comp=selected_comp,
        date_from=date_from,
        date_to=date_to
    )

    # Apply location filtering for match list
    if location_option and not df_sidebar.empty and location_option != "All":
        if "is_home" not in df_sidebar.columns:
            df_sidebar = add_derived_columns(df_sidebar)
        if location_option == "Home":
            df_sidebar = df_sidebar[df_sidebar["is_home"] == True].copy()
        elif location_option == "Away":
            df_sidebar = df_sidebar[df_sidebar["is_home"] == False].copy()

    # Build per-row match keys (preserve duplicates) => "Match Name | YYYY-MM-DD"
    def _row_match_key_series(r):
        m = str(r.get("match", "") or "")
        dval = r.get(DATE_COL, None)
        if isinstance(dval, (datetime, date)):
            ds = dval.strftime("%Y-%m-%d")
        else:
            try:
                ds = str(pd.to_datetime(dval).date())
            except Exception:
                ds = ""
        return f"{m} | {ds}" if ds else m

    if not df_sidebar.empty:
        row_keys = df_sidebar.apply(_row_match_key_series, axis=1).astype(str).tolist()
    else:
        row_keys = []

    match_options = ["All"] + row_keys
    selected_matches = st.multiselect(
        "Select match record(s) (choose 'All' to include all rows) - duplicates preserved",
        options=match_options,
        default=["All"],
        key="match_filter"
    )

    st.markdown("---")

    # Visualizations: add select-all toggle and multiselect
    st.markdown('<h2 style="color:#e7eefc; font-weight:900;">Visualizations</h2>', unsafe_allow_html=True)
    vis_options = list(VISUALS.keys())
    select_all_vis = st.checkbox("Select all visuals", value=True, help="Toggle to (de)select all visuals")
    vis_selected = st.multiselect(
        "Select one or more visuals",
        options=vis_options,
        default=vis_options if select_all_vis else [],
        key="vis_selected"
    )
    st.caption("Use the toggle to select all visuals, or pick specific visuals.")

    st.markdown("---")
    include_table = st.checkbox("Include data table in PDF", value=True)
    if st.button("Clear Filters"):
        st.experimental_rerun()

# =============== DATA FETCH + LOCAL FILTER ===============
# If df_sidebar exists (populated in sidebar), reuse it; otherwise fetch now.
try:
    df_all  # type: ignore
except NameError:
    df_all = fetch_match_data(
        selected_comp=selected_comp,
        date_from=date_from,
        date_to=date_to
    )

# Ensure location filter applied (in case df_all came from fetch here)
if location_option and not df_all.empty and location_option != "All":
    if "is_home" not in df_all.columns:
        df_all = add_derived_columns(df_all)
    if location_option == "Home":
        df_all = df_all[df_all["is_home"] == True].copy()
    elif location_option == "Away":
        df_all = df_all[df_all["is_home"] == False].copy()

# Build the same per-row key on df_all so selections map 1:1 to rows
def _row_match_key_series_r(r):
    m = str(r.get("match", "") or "")
    dval = r.get(DATE_COL, None)
    if isinstance(dval, (datetime, date)):
        ds = dval.strftime("%Y-%m-%d")
    else:
        try:
            ds = str(pd.to_datetime(dval).date())
        except Exception:
            ds = ""
    return f"{m} | {ds}" if ds else m

if not df_all.empty:
    df_all["_match_key"] = df_all.apply(_row_match_key_series_r, axis=1).astype(str)
else:
    df_all["_match_key"] = []

# Apply selected match record(s) from sidebar (match-key aware)
if not selected_matches or "All" in selected_matches:
    match_data_df = df_all.copy()
else:
    # selected_matches contains the per-row keys
    match_data_df = df_all[df_all["_match_key"].isin(selected_matches)].copy()

# =============== SUMMARY & TABLE ===============
st.markdown('<div class="fin-head">', unsafe_allow_html=True)
st.markdown(f'<span class="line">Found {len(match_data_df)} Matches</span>', unsafe_allow_html=True)
st.markdown('<span class="badge">Sunderland U21</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Removed on-page match details table per request (keeps KPIs below)
# If you want the raw table available, we can add an expander instead.

# KPI row (safe checks)
kpi_cols = st.columns(3)
with kpi_cols[0]:
    if "goals" in match_data_df.columns:
        st.metric("Total Goals", f"{int(pd.to_numeric(match_data_df['goals'], errors='coerce').fillna(0).sum())}")
with kpi_cols[1]:
    if "percentage_of_possession" in match_data_df.columns:
        st.metric("Avg Possession", f"{pd.to_numeric(match_data_df['percentage_of_possession'], errors='coerce').mean():.1f}%")
with kpi_cols[2]:
    if "ppda" in match_data_df.columns:
        st.metric("Avg PPDA", f"{pd.to_numeric(match_data_df['ppda'], errors='coerce').mean():.1f}")

# =============== RENDER VISUALS + COLLECT FOR PDF ===============
radar_pngs: List[bytes] = []
chart_pngs: List[bytes] = []

st.markdown('<h3 class="hsec" style="margin-top:10px;">Visualizations</h3>', unsafe_allow_html=True)

# Build list of match records to render radars for (preserve each row; do not de-duplicate)
if not selected_matches or "All" in selected_matches:
    # preserve order and every row
    match_records = match_data_df.reset_index().to_dict('records')
else:
    # use the per-row _match_key (match | YYYY-MM-DD) to select exact records
    sub = match_data_df[match_data_df["_match_key"].isin(selected_matches)].reset_index()
    match_records = sub.to_dict('records')

def chunked_records(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# Render radars for each selected match record in a 3-column grid on the site (center last row)
imgs_for_pdf = []
cols_per_row = 3

for row_chunk in chunked_records(match_records, cols_per_row):
    cols = st.columns(cols_per_row)
    k = len(row_chunk)
    if k == 3:
        positions = [0,1,2]
    elif k == 2:
        positions = [0,1]
    else:  # k == 1
        positions = [0]

    for pos, rec_idx in zip(positions, range(k)):
        rec = row_chunk[rec_idx]
        # build a one-row DataFrame from the record (preserves original row values)
        single_row_df = pd.DataFrame([rec])

        # Try to get a friendly caption (match + date if available)
        mname = str(rec.get("match", "") or "")
        if DATE_COL in rec and rec.get(DATE_COL) not in (None, ""):
            try:
                mdate = _safe_date(rec.get(DATE_COL))
                if mdate:
                    mname = f"{mname} | {mdate.strftime('%Y-%m-%d')}"
            except Exception:
                pass

        # render radar (or placeholder if nothing to plot)
        radar_img = chart_radar_for_match(single_row_df, df_all if not df_all.empty else match_data_df)
        if not radar_img:
            radar_img = chart_placeholder_radar(mname)

        with cols[pos]:
            st.image(radar_img, caption=f"Performance Radar — {mname}", use_container_width=True)

        imgs_for_pdf.append(radar_img)

# keep radars separate for PDF (all radars will be placed on one PDF page)
radar_pngs.extend(imgs_for_pdf)

# Preferred display order for other visuals (user-facing)
preferred_order = [
    "Match Overview (xG · Possession)",
    "Goals vs xG (Team)",
    "Rolling Averages (by window)",
    "PPDA Trend (Team)",
    "Duel Win % (Team)",
    "Final Third Activity (Team)",
    "Shots Against (Team)",
]

# Render other selected visuals in preferred order
for vis_name in preferred_order:
    if vis_name not in vis_selected:
        continue
    fn = VISUALS.get(vis_name)
    if not fn:
        continue
    img = fn(match_data_df)
    if not img:
        st.warning(f"Skipped '{vis_name}' (required columns not found).")
        continue
    st.image(img, caption=vis_name, use_container_width=True)
    chart_pngs.append(img)

# Create PDF: all radars on single page first, then other charts
report_title = "Sunderland U21 Customized Dashboard | Match Stats"
pdf_bytes = create_pdf_report(match_data_df, report_title, LOGO_PATH, radar_pngs, chart_pngs, include_table)

st.download_button(
    label="📄 Download Selected Visuals as PDF",
    data=pdf_bytes,
    file_name=f'sunderland_u21_report_{datetime.now().strftime("%Y-%m-%d")}.pdf',
    mime='application/pdf',
    key='download_pdf'
)
