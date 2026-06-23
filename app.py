import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLB DFS Stack Command Center", layout="wide")

TEAM_ALIAS = {
    "WSH": "WAS", "WAS": "WAS",
    "TB": "TBR", "TBR": "TBR",
    "SF": "SFG", "SFG": "SFG",
    "CWS": "CHW", "CHW": "CHW",
    "KC": "KCR", "KCR": "KCR",
    "SD": "SDP", "SDP": "SDP",
    "ATH": "OAK", "OAK": "OAK",
    "AZ": "ARI", "ARI": "ARI",
    "LAD": "LAD", "NYY": "NYY", "CIN": "CIN", "MIL": "MIL", "ATL": "ATL",
    "CLE": "CLE", "HOU": "HOU", "MIA": "MIA", "TEX": "TEX", "DET": "DET",
    "STL": "STL", "BAL": "BAL", "BOS": "BOS", "COL": "COL", "PIT": "PIT",
    "LAA": "LAA", "MIN": "MIN", "SEA": "SEA", "TOR": "TOR", "PHI": "PHI",
    "NYM": "NYM", "CHC": "CHC",
}

PARK_TEAM_ALIAS = {
    "ARI": "ARI", "ATH": "OAK", "OAK": "OAK", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "CHW": "CHW", "DET": "DET",
    "HOU": "HOU", "KCR": "KCR", "KC": "KCR", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "PHI": "PHI", "PIT": "PIT",
    "SDP": "SDP", "SD": "SDP", "SEA": "SEA", "SFG": "SFG", "SF": "SFG", "STL": "STL",
    "TBR": "TBR", "TB": "TBR", "TEX": "TEX", "TOR": "TOR", "WAS": "WAS", "WSH": "WAS",
}

ROSTER_KEYS = ["C", "1B", "2B", "3B", "SS", "OF", "OF1", "OF2", "OF3", "UTIL"]
PITCHER_KEYS = ["P", "P1", "P2", "SP", "SP1", "SP2"]


def norm_team(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    return TEAM_ALIAS.get(s, s)


def scale_series(s, invert=False):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(50.0, index=s.index)
    lo, hi = s.min(), s.max()
    if hi == lo:
        out = pd.Series(50.0, index=s.index)
    else:
        out = (s - lo) / (hi - lo) * 100
    if invert:
        out = 100 - out
    return out.fillna(50)


def read_csv(upload, fallback=None):
    if upload is not None:
        return pd.read_csv(upload)
    if fallback and Path(fallback).exists():
        return pd.read_csv(fallback)
    return None


def parse_dk_games(dk):
    dk = dk.copy()
    dk["Team"] = dk["TeamAbbrev"].map(norm_team)
    game_match = dk["Game Info"].astype(str).str.extract(r"([A-Z]{2,3})@([A-Z]{2,3})")
    dk["AwayRaw"] = game_match[0]
    dk["HomeRaw"] = game_match[1]
    dk["Away"] = dk["AwayRaw"].map(norm_team)
    dk["Home"] = dk["HomeRaw"].map(norm_team)
    dk["Opponent"] = np.where(dk["Team"].eq(dk["Away"]), dk["Home"], dk["Away"])
    dk["IsHome"] = dk["Team"].eq(dk["Home"])
    dk["ParkTeam"] = dk["Home"]
    game_cols = dk[["Game Info", "Away", "Home", "ParkTeam"]].drop_duplicates().reset_index(drop=True)
    team_game = dk[["Team", "Opponent", "IsHome", "ParkTeam", "Game Info"]].drop_duplicates("Team")
    return dk, game_cols, team_game


def clean_park(park):
    park = park.copy()
    team_col = "Names" if "Names" in park.columns else None
    if team_col is None:
        team_col = next((c for c in park.columns if "team" in c.lower() or "abbr" in c.lower()), park.columns[0])
    park["ParkTeam"] = park[team_col].map(lambda x: PARK_TEAM_ALIAS.get(str(x).strip().upper(), str(x).strip().upper()))
    if "Set" in park.columns:
        overall = park[park["Set"].astype(str).str.lower().eq("overall")]
        if not overall.empty:
            park = overall.copy()
    keep = ["ParkTeam"] + [c for c in ["Runs", "HR", "1B", "2B"] if c in park.columns]
    return park[keep].drop_duplicates("ParkTeam")


def build_stack_table(scoring, matchups, park, dk):
    scoring = scoring.copy()
    scoring["Team"] = scoring["names"].map(norm_team)
    slate_teams = scoring["Team"].dropna().unique().tolist()

    dk_players, games, team_game = parse_dk_games(dk)

    m = matchups.copy()
    if "Team" in m.columns:
        m["Team"] = m["Team"].map(norm_team)
    else:
        team_col = next((c for c in m.columns if c.lower() in ["names", "name", "teamabbrev", "team"]), None)
        if team_col:
            m["Team"] = m[team_col].map(norm_team)
    if "Opp" in m.columns:
        m["Opp"] = m["Opp"].map(norm_team)

    p = clean_park(park)

    table = scoring.merge(team_game, on="Team", how="left")
    table = table.merge(m, on="Team", how="left", suffixes=("", "_match"))
    table = table.merge(p, on="ParkTeam", how="left")

    table["ScoringProb"] = (
        0.35 * scale_series(table.get("avgScore")) +
        0.35 * scale_series(table.get("topScore")) +
        0.20 * scale_series(table.get("eightPlusRuns")) +
        0.10 * scale_series(table.get("avgFifthInning"))
    )

    table["TrendScore"] = (
        0.30 * scale_series(table.get("Trending Score")) +
        0.25 * scale_series(table.get("Trending 8+ For")) +
        0.25 * scale_series(table.get("Matchup Avg Trend")) +
        0.20 * scale_series(table.get("Matchup 8+ Trend"))
    )

    table["ParkScore"] = (
        0.40 * scale_series(table.get("Runs")) +
        0.40 * scale_series(table.get("HR")) +
        0.10 * scale_series(table.get("1B")) +
        0.10 * scale_series(table.get("2B"))
    )

    own = pd.to_numeric(table.get("teamOwnPct"), errors="coerce").replace(0, np.nan)
    top = pd.to_numeric(table.get("topScore"), errors="coerce")
    table["LeverageRatio"] = (top * 100 / own).replace([np.inf, -np.inf], np.nan).fillna(0)
    table["LeverageScore"] = 0.65 * scale_series(table["LeverageRatio"]) + 0.35 * scale_series(table.get("teamOwnPct"), invert=True)

    table["Raw Stack Score"] = (0.70 * table["ScoringProb"] + 0.20 * table["ParkScore"] + 0.10 * table["TrendScore"])
    table["GPP Stack Score"] = (0.40 * table["ScoringProb"] + 0.25 * table["TrendScore"] + 0.20 * table["ParkScore"] + 0.15 * table["LeverageScore"])
    table["Boom Score"] = (0.45 * scale_series(table.get("eightPlusRuns")) + 0.40 * scale_series(table.get("topScore")) + 0.15 * table["ParkScore"])
    table["Safe Stack Score"] = (0.45 * scale_series(table.get("avgScore")) + 0.30 * scale_series(table.get("winPercentage")) + 0.15 * scale_series(table.get("avgFifthInning")) + 0.10 * table["ParkScore"])
    table["Fade Risk Score"] = (0.65 * scale_series(table.get("teamOwnPct")) + 0.35 * scale_series(table.get("topScore"), invert=True))

    def tag(row):
        if row["GPP Stack Score"] >= 82 and row["Raw Stack Score"] >= 70:
            return "Elite Stack"
        if row["GPP Stack Score"] >= 72:
            return "Strong GPP Stack"
        if row["LeverageScore"] >= 70 and row["Boom Score"] >= 55:
            return "Leverage Stack"
        if row["Fade Risk Score"] >= 75 and row["Raw Stack Score"] < 60:
            return "Over-Owned Risk"
        if row["GPP Stack Score"] < 35:
            return "Avoid / Thin MME"
        return "Playable"

    table["Stack Label"] = table.apply(tag, axis=1)
    table = table[table["Team"].isin(slate_teams)].copy()
    return table, dk_players, games


def clean_player_name(x):
    """Clean DK export cells like 'Juan Soto (123456)' down to names."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def build_player_team_map(dk_players):
    name_cols = [c for c in ["Name", "Name + ID", "Player", "Player Name"] if c in dk_players.columns]
    if not name_cols:
        return {}
    out = {}
    for _, r in dk_players.iterrows():
        team = r.get("Team")
        for c in name_cols:
            nm = clean_player_name(r.get(c))
            if nm:
                out[nm.lower()] = team
    return out


def portfolio_position_columns(portfolio):
    """Find DK lineup player columns. Handles duplicate names like P/P.1/OF/OF.1/OF.2."""
    cols = []
    for c in portfolio.columns:
        base = re.sub(r"\.\d+$", "", str(c)).strip().upper()
        if base in ROSTER_KEYS + PITCHER_KEYS:
            cols.append(c)
    if cols:
        return cols
    # Fallback: columns that look like roster slots or contain player text.
    return [c for c in portfolio.columns if str(c).strip().upper() in ROSTER_KEYS + PITCHER_KEYS]


def hitter_position_columns(portfolio):
    cols = []
    for c in portfolio_position_columns(portfolio):
        base = re.sub(r"\.\d+$", "", str(c)).strip().upper()
        if base not in PITCHER_KEYS:
            cols.append(c)
    return cols


def detect_portfolio_stacks(portfolio, dk_players, stack_table=None):
    """Detect primary/secondary stacks directly from the uploaded portfolio CSV."""
    portfolio = portfolio.copy()
    player_team = build_player_team_map(dk_players)
    hit_cols = hitter_position_columns(portfolio)

    if not hit_cols:
        raise ValueError("Could not find hitter columns in portfolio CSV. Expected DK lineup columns like C, 1B, 2B, 3B, SS, OF, OF.1, OF.2, UTIL.")

    records = []
    for idx, row in portfolio.iterrows():
        team_counts = {}
        hitter_names = []
        for c in hit_cols:
            nm = clean_player_name(row.get(c))
            if not nm:
                continue
            hitter_names.append(nm)
            team = player_team.get(nm.lower())
            if pd.notna(team) and team:
                team_counts[team] = team_counts.get(team, 0) + 1

        sorted_counts = sorted(team_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        primary = sorted_counts[0][0] if sorted_counts else "Unknown"
        primary_n = sorted_counts[0][1] if sorted_counts else 0
        secondary = sorted_counts[1][0] if len(sorted_counts) > 1 else "None"
        secondary_n = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        counts_txt = ", ".join([f"{t}:{n}" for t, n in sorted_counts])
        stack_type = f"{primary_n}-{secondary_n}" if secondary_n else f"{primary_n}"

        records.append({
            "Lineup #": idx + 1,
            "Primary Stack": primary,
            "Primary Stack Size": primary_n,
            "Secondary Stack": secondary,
            "Secondary Stack Size": secondary_n,
            "Stack Type": stack_type,
            "Detected Team Counts": counts_txt,
        })

    detected = pd.concat([portfolio.reset_index(drop=True), pd.DataFrame(records)], axis=1)

    if stack_table is not None and not stack_table.empty:
        scores = stack_table[["Team", "GPP Stack Score", "Raw Stack Score", "Boom Score", "Fade Risk Score", "Stack Label"]].copy()
        detected = detected.merge(scores.add_prefix("Primary "), left_on="Primary Stack", right_on="Primary Team", how="left")
        detected = detected.merge(scores.add_prefix("Secondary "), left_on="Secondary Stack", right_on="Secondary Team", how="left")
        detected.drop(columns=[c for c in ["Primary Team", "Secondary Team"] if c in detected.columns], inplace=True)

        detected["Lineup Stack Score"] = (
            detected["Primary GPP Stack Score"].fillna(50) * 0.70 +
            detected["Secondary GPP Stack Score"].fillna(50) * 0.30
        )
        detected["Lineup Stack Grade"] = pd.cut(
            detected["Lineup Stack Score"],
            bins=[-1, 35, 55, 70, 82, 101],
            labels=["Cut", "Thin", "Playable", "Strong", "Elite"]
        ).astype(str)

    return detected




def build_pitcher_table(dk_players, stack_table):
    """Grade pitchers using DK info plus opponent stack weakness/cold trend/park risk."""
    dkp = dk_players.copy()
    if "Roster Position" not in dkp.columns:
        return pd.DataFrame()
    pitchers = dkp[dkp["Roster Position"].astype(str).str.upper().str.contains("P", na=False)].copy()
    if pitchers.empty:
        return pitchers

    pitchers["SalaryNum"] = pd.to_numeric(pitchers.get("Salary"), errors="coerce")
    pitchers["AvgFP"] = pd.to_numeric(pitchers.get("AvgPointsPerGame"), errors="coerce")
    pitchers["ValueScore"] = scale_series(pitchers["AvgFP"] / (pitchers["SalaryNum"] / 1000.0))
    pitchers["DKFormScore"] = scale_series(pitchers["AvgFP"])

    opp_context_cols = [
        "Team", "GPP Stack Score", "Raw Stack Score", "Boom Score", "Safe Stack Score", "Fade Risk Score",
        "ScoringProb", "TrendScore", "ParkScore", "avgScore", "eightPlusRuns", "topScore", "teamOwnPct", "Stack Label"
    ]
    opp_context_cols = [c for c in opp_context_cols if c in stack_table.columns]
    opp = stack_table[opp_context_cols].copy().add_prefix("Opp ")
    pitchers = pitchers.merge(opp, left_on="Opponent", right_on="Opp Team", how="left")

    # Better pitcher spot = opponent offense weaker + lower boom probability + less hitter-friendly park.
    pitchers["Opponent Weakness Score"] = (
        0.45 * scale_series(pitchers.get("Opp Raw Stack Score"), invert=True) +
        0.35 * scale_series(pitchers.get("Opp Boom Score"), invert=True) +
        0.20 * scale_series(pitchers.get("Opp TrendScore"), invert=True)
    )
    pitchers["Park Pitching Score"] = scale_series(pitchers.get("Opp ParkScore"), invert=True)
    pitchers["Pitcher Grade Score"] = (
        0.35 * pitchers["Opponent Weakness Score"] +
        0.25 * pitchers["DKFormScore"] +
        0.20 * pitchers["ValueScore"] +
        0.20 * pitchers["Park Pitching Score"]
    )

    def p_label(x):
        if x >= 82:
            return "Elite Pitching Spot"
        if x >= 70:
            return "Strong Pitching Spot"
        if x >= 55:
            return "Playable Pitcher"
        if x >= 40:
            return "Risky / GPP Only"
        return "Avoid Pitcher Spot"

    pitchers["Pitcher Grade"] = pitchers["Pitcher Grade Score"].apply(p_label)
    return pitchers


def pitcher_position_columns(portfolio):
    cols = []
    for c in portfolio_position_columns(portfolio):
        base = re.sub(r"\.\d+$", "", str(c)).strip().upper()
        if base in PITCHER_KEYS:
            cols.append(c)
    return cols


def build_pitcher_name_score_map(pitcher_table):
    out = {}
    if pitcher_table is None or pitcher_table.empty:
        return out
    name_cols = [c for c in ["Name", "Name + ID", "Player", "Player Name"] if c in pitcher_table.columns]
    for _, r in pitcher_table.iterrows():
        for c in name_cols:
            nm = clean_player_name(r.get(c))
            if nm:
                out[nm.lower()] = {
                    "score": r.get("Pitcher Grade Score", np.nan),
                    "grade": r.get("Pitcher Grade", ""),
                    "opp": r.get("Opponent", ""),
                }
    return out


def add_portfolio_pitcher_grades(graded, pitcher_table):
    """Attach P1/P2 pitcher grades and a combined lineup pitching grade to a portfolio."""
    graded = graded.copy()
    pcols = pitcher_position_columns(graded)
    if not pcols or pitcher_table is None or pitcher_table.empty:
        return graded
    pmap = build_pitcher_name_score_map(pitcher_table)

    for i, c in enumerate(pcols[:2], start=1):
        names = graded[c].map(clean_player_name)
        graded[f"P{i} Name"] = names
        graded[f"P{i} Grade Score"] = names.map(lambda n: pmap.get(n.lower(), {}).get("score", np.nan))
        graded[f"P{i} Grade"] = names.map(lambda n: pmap.get(n.lower(), {}).get("grade", "Unknown"))
        graded[f"P{i} Opp"] = names.map(lambda n: pmap.get(n.lower(), {}).get("opp", ""))

    score_cols = [c for c in ["P1 Grade Score", "P2 Grade Score"] if c in graded.columns]
    if score_cols:
        graded["Lineup Pitching Score"] = graded[score_cols].mean(axis=1)
        graded["Lineup Pitching Grade"] = pd.cut(
            graded["Lineup Pitching Score"],
            bins=[-1, 40, 55, 70, 82, 101],
            labels=["Bad", "Risky", "Playable", "Strong", "Elite"]
        ).astype(str)
        if "Lineup Stack Score" in graded.columns:
            graded["Overall Lineup Score"] = 0.70 * graded["Lineup Stack Score"].fillna(50) + 0.30 * graded["Lineup Pitching Score"].fillna(50)
    return graded


st.title("⚾ MLB DFS Stack Command Center")
st.caption("Scoring Sheet = slate master. DK Salaries = player/team/game map. Portfolio CSV = detected primary and secondary stacks.")

with st.sidebar:
    st.header("Upload CSVs")
    scoring_file = st.file_uploader("Scoring % Sheet", type="csv")
    matchups_file = st.file_uploader("Matchups Master Sheet", type="csv")
    park_file = st.file_uploader("Park Factors Sheet", type="csv")
    dk_file = st.file_uploader("DK Salaries Sheet", type="csv")
    portfolio_file = st.file_uploader("Optional: Lineup Portfolio Export", type="csv")
    st.divider()
    st.write("V3: Adds pitcher grades. Portfolio filters still come from the uploaded portfolio's detected stacks, not from all slate teams.")

fallbacks = {
    "scoring": "/mnt/data/MLB_Scoring_Pct_Draftkings_Main Slate (1).csv",
    "matchups": "/mnt/data/MLB_Matchups_export.csv",
    "park": "/mnt/data/park_factors (1).csv",
    "dk": "/mnt/data/DKSalaries (56).csv",
}

scoring = read_csv(scoring_file, fallbacks["scoring"])
matchups = read_csv(matchups_file, fallbacks["matchups"])
park = read_csv(park_file, fallbacks["park"])
dk = read_csv(dk_file, fallbacks["dk"])
portfolio = read_csv(portfolio_file) if portfolio_file is not None else None

if any(x is None for x in [scoring, matchups, park, dk]):
    st.info("Upload Scoring %, Matchups Master, Park Factors, and DK Salaries to begin.")
    st.stop()

try:
    stack_table, dk_players, games = build_stack_table(scoring, matchups, park, dk)
    pitcher_table = build_pitcher_table(dk_players, stack_table)
except Exception as e:
    st.error(f"Could not build rankings: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Stack Rankings", "Portfolio Mode", "Team Deep Dive", "Pitcher Grades"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slate Teams", stack_table["Team"].nunique())
    c2.metric("DK Players", len(dk_players))
    c3.metric("Games", len(games))
    c4.metric("Best GPP Stack", stack_table.sort_values("GPP Stack Score", ascending=False).iloc[0]["Team"])

    st.subheader("Slate Games / Park Mapping")
    st.dataframe(games, use_container_width=True, hide_index=True)

    view_cols = [
        "Team", "Opponent", "IsHome", "ParkTeam", "oppSP", "Stack Label",
        "GPP Stack Score", "Raw Stack Score", "Boom Score", "Safe Stack Score", "Fade Risk Score",
        "avgScore", "eightPlusRuns", "topScore", "teamOwnPct", "winPercentage",
        "Trending Score", "Trending 8+ For", "Matchup Avg Trend", "Runs", "HR", "1B", "2B", "LeverageRatio"
    ]
    view_cols = [c for c in view_cols if c in stack_table.columns]

    sort_choice = st.selectbox("Sort rankings by", ["GPP Stack Score", "Raw Stack Score", "Boom Score", "Safe Stack Score", "Fade Risk Score", "LeverageScore"])
    ranked = stack_table.sort_values(sort_choice, ascending=False)[view_cols]
    st.subheader("Stack Rankings")
    st.dataframe(ranked, use_container_width=True, hide_index=True)

    csv = ranked.to_csv(index=False).encode("utf-8")
    st.download_button("Download Stack Rankings CSV", csv, "mlb_stack_rankings.csv", "text/csv")

with tab2:
    st.subheader("Portfolio Mode — stacks detected from the portfolio CSV")
    if portfolio is None:
        st.info("Upload a Lineup Portfolio Export in the sidebar. Primary/secondary stack filter options will be created from that file only.")
    else:
        try:
            graded = detect_portfolio_stacks(portfolio, dk_players, stack_table)
            graded = add_portfolio_pitcher_grades(graded, pitcher_table)
        except Exception as e:
            st.error(f"Could not detect stacks from portfolio: {e}")
            st.stop()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lineups", len(graded))
        m2.metric("Primary Stacks", graded["Primary Stack"].replace("Unknown", np.nan).dropna().nunique())
        m3.metric("Secondary Stacks", graded["Secondary Stack"].replace("None", np.nan).dropna().nunique())
        m4.metric("Avg Lineup Stack Score", f"{graded.get('Lineup Stack Score', pd.Series([0])).mean():.1f}")

        primary_options = sorted([x for x in graded["Primary Stack"].dropna().unique().tolist() if x != "Unknown"])
        secondary_options = sorted([x for x in graded["Secondary Stack"].dropna().unique().tolist() if x not in ["None", "Unknown"]])
        stack_type_options = sorted(graded["Stack Type"].dropna().unique().tolist())

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            primary_filter = st.multiselect("Primary Stack", primary_options, default=[])
        with f2:
            secondary_filter = st.multiselect("Secondary Stack", secondary_options, default=[])
        with f3:
            stack_type_filter = st.multiselect("Stack Type", stack_type_options, default=[])
        with f4:
            grade_options = sorted(graded["Lineup Stack Grade"].dropna().unique().tolist()) if "Lineup Stack Grade" in graded.columns else []
            grade_filter = st.multiselect("Lineup Stack Grade", grade_options, default=[])

        filtered = graded.copy()
        if primary_filter:
            filtered = filtered[filtered["Primary Stack"].isin(primary_filter)]
        if secondary_filter:
            filtered = filtered[filtered["Secondary Stack"].isin(secondary_filter)]
        if stack_type_filter:
            filtered = filtered[filtered["Stack Type"].isin(stack_type_filter)]
        if grade_filter and "Lineup Stack Grade" in filtered.columns:
            filtered = filtered[filtered["Lineup Stack Grade"].isin(grade_filter)]

        st.write(f"Showing **{len(filtered)}** of **{len(graded)}** lineups")
        show_cols = [
            "Lineup #", "Primary Stack", "Primary Stack Size", "Secondary Stack", "Secondary Stack Size", "Stack Type",
            "Detected Team Counts", "Lineup Stack Score", "Lineup Stack Grade", "Lineup Pitching Score", "Lineup Pitching Grade", "Overall Lineup Score",
            "P1 Name", "P1 Grade Score", "P1 Grade", "P1 Opp", "P2 Name", "P2 Grade Score", "P2 Grade", "P2 Opp",
            "Primary GPP Stack Score", "Secondary GPP Stack Score", "Primary Stack Label", "Secondary Stack Label"
        ]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

        exposure = (
            graded.groupby(["Primary Stack", "Primary Stack Size"], dropna=False)
            .size().reset_index(name="Lineups")
            .sort_values("Lineups", ascending=False)
        )
        exposure["Exposure %"] = exposure["Lineups"] / len(graded) * 100
        if "Team" in stack_table.columns:
            exposure = exposure.merge(stack_table[["Team", "GPP Stack Score", "Stack Label"]], left_on="Primary Stack", right_on="Team", how="left").drop(columns=["Team"])
        st.subheader("Primary Stack Exposure — from portfolio CSV")
        st.dataframe(exposure, use_container_width=True, hide_index=True)

        out = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered portfolio CSV", out, "filtered_portfolio_with_detected_stacks.csv", "text/csv")

with tab3:
    view_cols = ["Team", "GPP Stack Score", "Raw Stack Score", "Boom Score", "Stack Label"]
    teams_for_select = stack_table.sort_values("GPP Stack Score", ascending=False)["Team"].tolist()
    team = st.selectbox("Choose a team", teams_for_select)
    row = stack_table[stack_table["Team"].eq(team)].iloc[0]

    left, right = st.columns([1, 2])
    with left:
        st.metric("GPP Stack Score", f"{row['GPP Stack Score']:.1f}")
        st.metric("Raw Stack Score", f"{row['Raw Stack Score']:.1f}")
        st.metric("Boom Score", f"{row['Boom Score']:.1f}")
        st.metric("Label", row["Stack Label"])
    with right:
        reasons = []
        reasons.append(f"Top-score chance: {row.get('topScore', np.nan):.2%}" if pd.notna(row.get('topScore')) else "Top-score chance unavailable")
        reasons.append(f"8+ run chance: {row.get('eightPlusRuns', np.nan):.2%}" if pd.notna(row.get('eightPlusRuns')) else "8+ run chance unavailable")
        reasons.append(f"Team ownership: {row.get('teamOwnPct', np.nan):.1f}%" if pd.notna(row.get('teamOwnPct')) else "Ownership unavailable")
        reasons.append(f"Park: {row.get('ParkTeam', 'N/A')} | Runs {row.get('Runs', np.nan):.2f}, HR {row.get('HR', np.nan):.2f}" if pd.notna(row.get('Runs')) else "Park factor unavailable")
        reasons.append(f"Trend score input: {row.get('Trending Score', np.nan):.2f}" if pd.notna(row.get('Trending Score')) else "Trend unavailable")
        st.write("**Why this team grades here:**")
        for r in reasons:
            st.write(f"- {r}")

    st.subheader("Hitters From Selected Stack")
    hitters = dk_players[(dk_players["Team"].eq(team)) & (~dk_players["Roster Position"].astype(str).str.contains("P", na=False))].copy()
    if hitters.empty:
        st.warning("No hitters found for this team in DK salaries.")
    else:
        hitter_cols = ["Name", "Roster Position", "Salary", "AvgPointsPerGame", "Game Info", "Team", "Opponent", "IsHome"]
        hitter_cols = [c for c in hitter_cols if c in hitters.columns]
        st.dataframe(hitters[hitter_cols].sort_values(["AvgPointsPerGame", "Salary"], ascending=False), use_container_width=True, hide_index=True)


with tab4:
    st.subheader("Pitcher Grades")
    st.caption("Pitcher score uses DK salary/Avg FP plus opponent stack weakness, opponent boom risk, trends, and park risk.")
    if pitcher_table is None or pitcher_table.empty:
        st.info("No pitchers found in the DK Salaries sheet.")
    else:
        pcols = [
            "Name", "Team", "Opponent", "IsHome", "Salary", "AvgPointsPerGame", "Pitcher Grade Score", "Pitcher Grade",
            "Opponent Weakness Score", "Park Pitching Score", "DKFormScore", "ValueScore",
            "Opp Raw Stack Score", "Opp Boom Score", "Opp Stack Label", "Game Info"
        ]
        pcols = [c for c in pcols if c in pitcher_table.columns]
        sort_p = st.selectbox("Sort pitchers by", ["Pitcher Grade Score", "Opponent Weakness Score", "Park Pitching Score", "ValueScore", "AvgPointsPerGame"])
        st.dataframe(pitcher_table.sort_values(sort_p, ascending=False)[pcols], use_container_width=True, hide_index=True)
        pcsv = pitcher_table[pcols].sort_values(sort_p, ascending=False).to_csv(index=False).encode("utf-8")
        st.download_button("Download Pitcher Grades CSV", pcsv, "mlb_pitcher_grades.csv", "text/csv")


st.caption("V3: Adds pitcher grades and portfolio pitching score. Portfolio stack filters still come only from the uploaded portfolio CSV.")
