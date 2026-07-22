from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from nba_api.stats.endpoints import leaguedashplayerstats

st.set_page_config(
    page_title="NBA Player Impact Analyzer",
    page_icon="🏀",
    layout="wide",
)

# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
# Main path for the cleaned GitHub project:
# project/
# ├── app/streamlit_app.py
# └── data/processed/player_profiles_final.csv
DATA_PATH = Path("/Users/Marcy_Student/Desktop/NBA project (gh)/notebooks/data/processed/player_profiles_final.csv")

# Backup paths in case you run this file from a different folder while testing.
FALLBACK_DATA_PATHS = [
    Path("data/processed/player_profiles_final.csv"),
    Path("player_profiles_final.csv"),
    Path(__file__).resolve().parent / "player_profiles_final.csv",
]

NBA_HEADSHOT_BASE_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760"

REQUIRED_COLUMNS = [
    "playerName",
    "personId",
    "games_played",
    "avg_minutes",
    "avg_fgm",
    "fg_pct",
    "ts_pct",
    "net_rating",
    "pie",
    "usg_pct",
    "pct_ast",
    "pct_reb",
    "impact_score",
    "offensive_creation",
    "lineup_defensive_impact",
    "spacing_value",
    "playmaking_value",
    "rebounding_value",
    "points_generated_by_assists",
    "lineup_score",
    "lineup_role",
    "scorer_profile",
    "defensive_profile",
]

PROFILE_COLUMNS = [
    "playerName",
    "team",
    "pos",
    "games_played",
    "avg_minutes",
    "impact_score",
    "lineup_score",
    "lineup_role",
    "scorer_profile",
    "defensive_profile",
    "profile_description",
    "defensive_description",
]

RANKING_COLUMNS = [
    "playerName",
    "team",
    "pos",
    "games_played",
    "avg_minutes",
    "impact_score",
    "lineup_score",
    "ts_pct",
    "usg_pct",
    "pie",
    "lineup_role",
]

MATCHUP_METRICS = {
    "Offense": "offensive_creation",
    "Defense": "lineup_defensive_impact",
    "Spacing": "spacing_value",
    "Playmaking": "playmaking_value",
    "Rebounding": "rebounding_value",
}


# -----------------------------------------------------------------------------
# Data loading and utility helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the final player profile dataset and add NBA headshot URLs."""
    path = DATA_PATH
    if not path.exists():
        for fallback_path in FALLBACK_DATA_PATHS:
            if fallback_path.exists():
                path = fallback_path
                break

    if not path.exists():
        raise FileNotFoundError(
            "Could not find player_profiles_final.csv. Put it in data/processed/ or beside this app file."
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=["playerName"]).copy()
    df["playerName"] = df["playerName"].astype(str).str.strip()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["personId"] = pd.to_numeric(df["personId"], errors="coerce").astype("Int64")

    # Build a reliable NBA headshot URL from personId.
    df["headshot_url"] = df["personId"].apply(
        lambda player_id: f"{NBA_HEADSHOT_BASE_URL}/{int(player_id)}.png" if pd.notna(player_id) else None
    )

    numeric_columns = [
        "games_played",
        "avg_minutes",
        "avg_fgm",
        "fg_pct",
        "ts_pct",
        "net_rating",
        "pie",
        "usg_pct",
        "pct_ast",
        "pct_reb",
        "impact_score",
        "offensive_creation",
        "lineup_defensive_impact",
        "spacing_value",
        "playmaking_value",
        "rebounding_value",
        "points_generated_by_assists",
        "lineup_score",
        "pct_stl",
        "pct_blk",
        "avg_shot_distance",
        "freq_rim",
        "freq_short_paint",
        "freq_mid",
        "freq_long_mid",
        "freq_3p",
        "fg_rim",
        "fg_short_paint",
        "fg_mid",
        "fg_long_mid",
        "fg_3p",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values("playerName").reset_index(drop=True)


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return only the columns that exist in the dataframe."""
    return [column for column in columns if column in df.columns]


def make_positive_marker_size(df: pd.DataFrame, column: str) -> pd.Series:
    """Plotly marker sizes must be zero or positive.

    Some basketball metrics, like impact_score or net_rating, can be negative.
    This helper keeps the real metric unchanged for analysis/hover text, but creates
    a safe positive copy for bubble sizes in scatter plots.
    """
    sizes = pd.to_numeric(df[column], errors="coerce").fillna(0)
    sizes = sizes.clip(lower=0)

    if sizes.max() == 0:
        return pd.Series(1, index=df.index)

    return sizes + 0.1


def format_decimal(value, digits: int = 3):
    if pd.isna(value):
        return "N/A"
    return round(float(value), digits)


def format_percent(value, digits: int = 1):
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def min_max_scale_value(value: float, series: pd.Series) -> float:
    clean_series = pd.to_numeric(series, errors="coerce").fillna(0)
    clean_value = 0 if pd.isna(value) else value

    if clean_series.max() == clean_series.min():
        return 0.0

    return float((clean_value - clean_series.min()) / (clean_series.max() - clean_series.min()))


def calculate_lineup_score(lineup: pd.DataFrame, all_players: pd.DataFrame) -> dict:
    """Calculate a normalized lineup score and category scores for one rotation."""
    metrics = {
        "offense": ("offensive_creation", 0.40),
        "defense": ("lineup_defensive_impact", 0.25),
        "spacing": ("spacing_value", 0.15),
        "playmaking": ("playmaking_value", 0.10),
        "rebounding": ("rebounding_value", 0.10),
    }

    scores = {}
    raw_values = {}

    for label, (column, weight) in metrics.items():
        raw_value = lineup[column].fillna(0).mean()
        scaled_value = min_max_scale_value(raw_value, all_players[column])
        scores[f"{label}_score"] = scaled_value
        raw_values[f"raw_{label}"] = raw_value

    lineup_score = sum(scores[f"{label}_score"] * weight for label, (_, weight) in metrics.items())
    return {"lineup_score": lineup_score, **scores, **raw_values}


def project_final_score(team_a_scores: dict, team_b_scores: dict) -> tuple[int, int]:
    """Convert the lineup score difference into a projected final score."""
    base_score = 112
    score_gap = team_a_scores["lineup_score"] - team_b_scores["lineup_score"]
    projected_margin = score_gap * 75

    team_a_points = base_score + projected_margin / 2
    team_b_points = base_score - projected_margin / 2

    team_a_points = max(85, min(140, team_a_points))
    team_b_points = max(85, min(140, team_b_points))

    return round(team_a_points), round(team_b_points)

def get_first_available_column(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

def create_projected_box_score(rotation: pd.DataFrame, target_team_points: int) -> pd.DataFrame:
    """Create a simple projected player box score for a selected rotation."""
    box = rotation.copy()
    box["MIN"] = np.where(box["rotation_role"] == "Starter", 34, 14)

    total_minutes = box["MIN"].sum()
    if total_minutes > 0:
        box["MIN"] = box["MIN"] * (240 / total_minutes)

    # -----------------------------
    # POINTS / SHOTS LOGIC
    # -----------------------------
    # Use shot-taking stats first, so high-volume scorers shoot more.
    # Fallback to offensive creation if shot-volume columns are missing.
    shot_volume_col = None

    for col in ["usg_pct", "usage_pct", "pct_usg", "fga", "FGA", "avg_fga", "field_goal_attempts"]:
        if col in box.columns:
            shot_volume_col = col
            break

    if shot_volume_col:
        box["shot_weight"] = box[shot_volume_col].fillna(0)
    else:
        box["shot_weight"] = (
            box["offensive_creation"].fillna(0) * 0.60
            + box["lineup_score"].fillna(0) * 20
        )

    # Keep weights from going negative
    box["shot_weight"] = box["shot_weight"].clip(lower=0)

    # Give starters a little more shot opportunity than bench players
    box["shot_weight"] = box["shot_weight"] * (box["MIN"] / box["MIN"].mean())

    if box["shot_weight"].sum() == 0:
        box["shot_weight"] = 1

    # Team FGA estimate
    team_fga = target_team_points / 1.12

    box["FGA"] = (box["shot_weight"] / box["shot_weight"].sum()) * team_fga

    # Use TS% to estimate points from shot attempts
    if "ts_pct" in box.columns:
        box["efficiency"] = box["ts_pct"].replace(0, np.nan).fillna(0.55)
    else:
        box["efficiency"] = 0.55

    box["efficiency"] = box["efficiency"].clip(lower=0.45, upper=0.70)

    box["PTS_raw"] = box["FGA"] * box["efficiency"] * 2.05

    if box["PTS_raw"].sum() > 0:
        box["PTS"] = (box["PTS_raw"] / box["PTS_raw"].sum()) * target_team_points
    else:
        box["PTS"] = target_team_points / len(box)

    box["FGM"] = box["PTS"] / 2.25

    # -----------------------------
    # ASSISTS
    # -----------------------------
    box["AST"] = box["points_generated_by_assists"].fillna(0)

    # No negative assists
    box["AST"] = box["AST"].clip(lower=0)

    box["AST"] = (box["AST"] / box["AST"].sum()) * 25 if box["AST"].sum() > 0 else 0

    # -----------------------------
    # REBOUNDS
    # -----------------------------
    box["REB"] = box["pct_reb"].fillna(0)

    # No negative rebounds
    box["REB"] = box["REB"].clip(lower=0)

    box["REB"] = (box["REB"] / box["REB"].sum()) * 44 if box["REB"].sum() > 0 else 0

    # -----------------------------
    # STEALS / BLOCKS
    # -----------------------------
    steal_col = None
    for col in ["stl", "STL", "steals", "pct_stl", "stl_pct", "steal_pct"]:
        if col in box.columns:
            steal_col = col
            break

    block_col = None
    for col in ["blk", "BLK", "blocks", "pct_blk", "blk_pct", "block_pct"]:
        if col in box.columns:
            block_col = col
            break

    if steal_col:
        box["STL_weight"] = box[steal_col].fillna(0)
    else:
        box["STL_weight"] = box["lineup_defensive_impact"].fillna(0)

    if block_col:
        box["BLK_weight"] = box[block_col].fillna(0)
    else:
        box["BLK_weight"] = (
            box["lineup_defensive_impact"].fillna(0) * 0.70
            + box["rebounding_value"].fillna(0) * 0.30
        )

    # This is the fix for negative steals/blocks
    box["STL_weight"] = box["STL_weight"].clip(lower=0)
    box["BLK_weight"] = box["BLK_weight"].clip(lower=0)

    if box["STL_weight"].sum() > 0:
        box["STL"] = (box["STL_weight"] / box["STL_weight"].sum()) * 7
    else:
        box["STL"] = 7 / len(box)

    if box["BLK_weight"].sum() > 0:
        box["BLK"] = (box["BLK_weight"] / box["BLK_weight"].sum()) * 5
    else:
        box["BLK"] = 5 / len(box)

    # Extra safety: no negative box score stats
    for col in ["PTS", "REB", "AST", "STL", "BLK", "FGM", "FGA"]:
        if col in box.columns:
            box[col] = box[col].clip(lower=0)

    # -----------------------------
    # ROUNDING
    # -----------------------------
    box["MIN"] = box["MIN"].round(1)

    for column in ["PTS", "AST", "REB", "STL", "BLK", "FGM", "FGA"]:
        if column in box.columns:
            box[column] = box[column].round(0).astype(int)

    display_columns = [
        "rotation_role",
        "playerName",
        "team",
        "pos",
        "lineup_role",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "FGM",
        "FGA",
        "scorer_profile",
        "defensive_profile",
    ]

    display_columns = available_columns(box, display_columns)

    return box[display_columns]


def create_team_box_score(player_box: pd.DataFrame) -> pd.DataFrame:
    """Summarize a player box score into team totals."""
    team_box = pd.DataFrame([
        {
            "PTS": player_box["PTS"].sum(),
            "REB": player_box["REB"].sum(),
            "AST": player_box["AST"].sum(),
            "STL": player_box["STL"].sum(),
            "BLK": player_box["BLK"].sum(),
            "FGM": player_box["FGM"].sum(),
            "FGA": player_box["FGA"].sum(),
        }
    ])

    team_box["FG%"] = (team_box["FGM"] / team_box["FGA"] * 100).replace(0, np.nan)
    return team_box.round(1)


def show_player_card(player: pd.Series, image_width: int = 115) -> None:
    """Display a small player card with image, name, role, and score."""
    if pd.notna(player.get("headshot_url")):
        st.image(player["headshot_url"], width=image_width)

    st.markdown(f"**{player['playerName']}**")
    if "team" in player and pd.notna(player.get("team")):
        st.caption(f"{player.get('team')} | {player.get('pos', 'N/A')}")
    st.caption(player.get("lineup_role", "N/A"))
    st.caption(f"Lineup score: {player.get('lineup_score', 0):.3f}")


def show_duplicate_player_warning(*selected_groups: list[str]) -> None:
    selected_players = [player for group in selected_groups for player in group]
    duplicates = sorted({player for player in selected_players if selected_players.count(player) > 1})

    if duplicates:
        st.warning(
            "Duplicate player warning: "
            + ", ".join(duplicates)
            + " was selected more than once. The app will still run, but the matchup may be less realistic."
        )


def select_team(team_label: str, player_list: list[str]) -> tuple[list[str], list[str]]:
    st.subheader(team_label)

    starters = st.multiselect(
        f"Select {team_label} Starting 5",
        player_list,
        max_selections=5,
        key=f"{team_label}_starters",
    )

    remaining_players = [player for player in player_list if player not in starters]
    bench = st.multiselect(
        f"Select {team_label} Bench Players (optional, up to 5)",
        remaining_players,
        max_selections=5,
        key=f"{team_label}_bench",
    )

    return starters, bench


def build_rotation(players_df: pd.DataFrame, starters: list[str], bench: list[str]) -> pd.DataFrame:
    starting_df = players_df[players_df["playerName"].isin(starters)].copy()
    bench_df = players_df[players_df["playerName"].isin(bench)].copy()

    return pd.concat(
        [
            starting_df.assign(rotation_role="Starter"),
            bench_df.assign(rotation_role="Bench"),
        ],
        ignore_index=True,
    )


def show_rotation(rotation: pd.DataFrame, title: str) -> None:
    st.subheader(title)

    starters = rotation[rotation["rotation_role"] == "Starter"].reset_index(drop=True)
    if not starters.empty:
        starter_columns = st.columns(5)
        for index, (_, player) in enumerate(starters.iterrows()):
            with starter_columns[index]:
                show_player_card(player)

    bench = rotation[rotation["rotation_role"] == "Bench"].reset_index(drop=True)
    if not bench.empty:
        st.caption("Bench")
        bench_columns = st.columns(5)
        for index, (_, player) in enumerate(bench.iterrows()):
            with bench_columns[index]:
                show_player_card(player)


def show_matchup_breakdown(team_a_scores: dict, team_b_scores: dict) -> None:
    comparison_df = pd.DataFrame(
        {
            "Category": ["Offense", "Defense", "Spacing", "Playmaking", "Rebounding"],
            "Team A": [
                team_a_scores["offense_score"],
                team_a_scores["defense_score"],
                team_a_scores["spacing_score"],
                team_a_scores["playmaking_score"],
                team_a_scores["rebounding_score"],
            ],
            "Team B": [
                team_b_scores["offense_score"],
                team_b_scores["defense_score"],
                team_b_scores["spacing_score"],
                team_b_scores["playmaking_score"],
                team_b_scores["rebounding_score"],
            ],
        }
    )

    comparison_long = comparison_df.melt(
        id_vars="Category",
        var_name="Team",
        value_name="Score",
    )

    fig = px.bar(
        comparison_long,
        x="Category",
        y="Score",
        color="Team",
        barmode="group",
        title="Lineup Strength Comparison",
        range_y=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)


def build_custom_lineup_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Create a weighted score using scaled lineup category values."""
    score = pd.Series(0.0, index=df.index)
    total_weight = sum(weights.values())

    if total_weight == 0:
        return score

    for label, weight in weights.items():
        column = MATCHUP_METRICS[label]
        scaled = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        score += scaled.fillna(0) * (weight / total_weight)

    return score


# -----------------------------------------------------------------------------
# Tab sections
# -----------------------------------------------------------------------------
@st.cache_data(ttl=21600)
def load_mvp_rankings(local_df: pd.DataFrame) -> pd.DataFrame:
    """Load current-season NBA stats if possible. If NBA API fails, use local data."""
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            timeout=60,
        ).get_data_frames()[0]

        stats = stats[
            (stats["GP"] >= 20) &
            (stats["MIN"] >= 25)
        ].copy()

        stats["mvp_score"] = (
            stats["PTS"] * 1.0 +
            stats["REB"] * 1.2 +
            stats["AST"] * 1.5 +
            stats["STL"] * 3 +
            stats["BLK"] * 3 +
            stats["PLUS_MINUS"] * 0.7 +
            stats["W_PCT"] * 20
        )

        stats["mvp_score"] = stats["mvp_score"].round(1)

        stats["headshot_url"] = stats["PLAYER_ID"].apply(
            lambda player_id: f"{NBA_HEADSHOT_BASE_URL}/{int(player_id)}.png"
        )

        stats["source"] = "NBA API"

        return stats.sort_values("mvp_score", ascending=False)

    except Exception:
        fallback = local_df.copy()

        fallback["mvp_score"] = (
            fallback["impact_score"].fillna(0) * 0.35 +
            fallback["lineup_score"].fillna(0) * 100 * 0.25 +
            fallback["offensive_creation"].fillna(0) * 0.20 +
            fallback["lineup_defensive_impact"].fillna(0) * 0.20
        )

        fallback["mvp_score"] = fallback["mvp_score"].round(1)

        fallback = fallback.rename(columns={
            "playerName": "PLAYER_NAME",
            "team": "TEAM_ABBREVIATION",
            "games_played": "GP",
            "avg_minutes": "MIN",
            "pct_reb": "REB",
            "pct_ast": "AST",
            "pct_stl": "STL",
            "pct_blk": "BLK",
            "net_rating": "PLUS_MINUS"
        })

        if "PTS" not in fallback.columns:
            fallback["PTS"] = fallback["avg_fgm"].fillna(0) * 2

        if "W_PCT" not in fallback.columns:
            fallback["W_PCT"] = np.nan

        fallback["source"] = "Local CSV fallback"

        return fallback.sort_values("mvp_score", ascending=False)

def show_overview_tab(df: pd.DataFrame) -> None:
    st.header("Project Overview")
    st.write(
        "This app turns player-level NBA data into a scouting-style dashboard. "
        "It ranks players, finds efficient underused players, compares player profiles, "
        "and simulates custom team matchups."
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Players", f"{len(df):,}")
    metric_col2.metric("Avg Impact Score", format_decimal(df["impact_score"].mean(), 2))
    metric_col3.metric("Avg TS%", format_percent(df["ts_pct"].mean(), 1))
    metric_col4.metric("Avg Usage", format_percent(df["usg_pct"].mean(), 1))

    st.subheader("Current Season MVP Predictor")

    try:
        mvp_df = load_mvp_rankings(df)
        top_mvp = mvp_df.iloc[0]

        mvp_col1, mvp_col2 = st.columns([1, 3])

        with mvp_col1:
            st.image(top_mvp["headshot_url"], width=150)

        with mvp_col2:
            st.markdown(f"### MVP Leader: {top_mvp['PLAYER_NAME']}")
            st.write(f"**Team:** {top_mvp['TEAM_ABBREVIATION']}")
            st.write(f"**MVP Score:** {top_mvp['mvp_score']}")
            st.write(
                f"**Stats:** {top_mvp['PTS']:.1f} PPG, "
                f"{top_mvp['REB']:.1f} RPG, "
                f"{top_mvp['AST']:.1f} APG"
            )

        st.dataframe(
            mvp_df[
                [
                    "PLAYER_NAME",
                    "TEAM_ABBREVIATION",
                    "mvp_score",
                    "PTS",
                    "REB",
                    "AST",
                    "STL",
                    "BLK",
                    "W_PCT",
                    "PLUS_MINUS",
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as error:
        st.warning("MVP stats could not load.")
        st.caption(str(error))

    st.divider()
    
    st.subheader("What the app does")
    st.markdown(
        """
        - **Player Rankings:** sorts players by impact, efficiency, usage, and lineup value.
        - **Hidden Gems:** finds players with strong efficiency, lower usage, and above-average impact.
        - **Player Profiles:** shows a single player's scoring, defense, shot profile, and lineup fit.
        - **Lineup Builder:** creates a suggested lineup based on the type of team you want to build.
        - **Matchup Simulator:** lets you build two rotations and projects a final score and box score.
        - **Visualizations:** explores relationships between scoring, efficiency, usage, and lineup impact.
        """
    )

    st.subheader("Lineup score formula")
    formula_df = pd.DataFrame(
        {
            "Category": ["Offense", "Defense", "Spacing", "Playmaking", "Rebounding"],
            "Weight": ["40%", "25%", "15%", "10%", "10%"],
            "Purpose": [
                "Measures scoring and offensive creation",
                "Captures defensive value and role impact",
                "Rewards shooting gravity and floor spacing",
                "Captures passing and assist creation",
                "Rewards possession value through rebounding",
            ],
        }
    )
    st.dataframe(formula_df, use_container_width=True, hide_index=True)

    st.info(
        "This is a portfolio analytics project, not a betting model. The projections are simplified estimates based on player profile metrics."
    )


def show_rankings_tab(df: pd.DataFrame) -> None:
    st.header("Player Rankings")
    st.write("Sort players by impact score, lineup score, efficiency, usage, or other profile metrics.")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        sort_options = [
            "impact_score",
            "lineup_score",
            "offensive_creation",
            "lineup_defensive_impact",
            "spacing_value",
            "playmaking_value",
            "rebounding_value",
            "ts_pct",
            "usg_pct",
            "pie",
        ]
        sort_by = st.selectbox("Rank by", sort_options, index=0)

    with filter_col2:
        min_games = st.slider(
            "Minimum games played",
            min_value=0,
            max_value=int(df["games_played"].max()),
            value=20,
        )

    with filter_col3:
        top_n = st.slider("Number of players", min_value=5, max_value=50, value=15)

    filtered = df[df["games_played"] >= min_games].copy()
    ranked = filtered.sort_values(sort_by, ascending=False).head(top_n)

    st.subheader(f"Top {top_n} Players by {sort_by.replace('_', ' ').title()}")
    st.dataframe(
        ranked[available_columns(ranked, RANKING_COLUMNS)].round(3),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        ranked.sort_values(sort_by, ascending=True),
        x=sort_by,
        y="playerName",
        orientation="h",
        hover_data=available_columns(ranked, ["team", "pos", "lineup_role", "ts_pct", "usg_pct"]),
        title=f"Top Players by {sort_by.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_hidden_gems_tab(df: pd.DataFrame) -> None:
    st.header("Hidden Gems")
    st.write(
        "Hidden gems are players with above-median efficiency, below-median usage, and above-median impact score. "
        "This helps find players who are productive without needing the ball as much."
    )

    hidden_gems = df[
        (df["ts_pct"] > df["ts_pct"].median())
        & (df["usg_pct"] < df["usg_pct"].median())
        & (df["impact_score"] > df["impact_score"].median())
    ].copy()

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of hidden gems", min_value=5, max_value=30, value=10)
    with col2:
        sort_by = st.selectbox(
            "Sort hidden gems by",
            ["impact_score", "ts_pct", "lineup_score", "pie"],
            index=0,
            key="hidden_gem_sort",
        )

    hidden_gems = hidden_gems.sort_values(sort_by, ascending=False).head(top_n)

    st.dataframe(
        hidden_gems[available_columns(hidden_gems, RANKING_COLUMNS)].round(3),
        use_container_width=True,
        hide_index=True,
    )

    scatter_df = df.copy()
    scatter_df["bubble_size"] = make_positive_marker_size(scatter_df, "impact_score")

    fig = px.scatter(
        scatter_df,
        x="usg_pct",
        y="ts_pct",
        size="bubble_size",
        color="lineup_role" if "lineup_role" in scatter_df.columns else None,
        hover_name="playerName",
        hover_data=available_columns(scatter_df, ["team", "pos", "impact_score", "lineup_score"]),
        title="Usage vs Efficiency",
    )
    fig.add_vline(x=df["usg_pct"].median(), line_dash="dash")
    fig.add_hline(y=df["ts_pct"].median(), line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)


def show_player_profiles_tab(df: pd.DataFrame) -> None:
    st.header("Player Profiles")
    st.write("Search one player and view their scoring profile, defensive profile, shot profile, and lineup fit.")

    player_list = sorted(df["playerName"].dropna().unique())
    default_index = player_list.index("LeBron James") if "LeBron James" in player_list else 0
    selected_player = st.selectbox("Choose a player", player_list, index=default_index)
    player = df[df["playerName"] == selected_player].iloc[0]

    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        if pd.notna(player.get("headshot_url")):
            st.image(player["headshot_url"], width=190)
    with header_col2:
        st.subheader(player["playerName"])
        team = player.get("team", "N/A")
        position = player.get("pos", "N/A")
        st.caption(f"{team} | {position} | {player.get('lineup_role', 'N/A')}")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Impact Score", format_decimal(player.get("impact_score"), 2))
        metric_col2.metric("Lineup Score", format_decimal(player.get("lineup_score"), 3))
        metric_col3.metric("TS%", format_percent(player.get("ts_pct"), 1))
        metric_col4.metric("Usage", format_percent(player.get("usg_pct"), 1))

    st.subheader("Profile Summary")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown("**Scoring Profile**")
        st.write(player.get("scorer_profile", "N/A"))
        st.write(player.get("profile_description", ""))
    with summary_col2:
        st.markdown("**Defensive Profile**")
        st.write(player.get("defensive_profile", "N/A"))
        st.write(player.get("defensive_description", ""))

    st.subheader("Player Metrics vs League Average")

    metric_map = {
        "Offense": "offensive_creation",
        "Defense": "lineup_defensive_impact",
        "Spacing": "spacing_value",
        "Playmaking": "playmaking_value",
        "Rebounding": "rebounding_value",
    }

    metric_rows = []

    for metric_name, column_name in metric_map.items():
        league_min = df[column_name].min()
        league_max = df[column_name].max()
        league_avg_raw = df[column_name].mean()
        player_raw = player.get(column_name, np.nan)

        if league_max == league_min:
            player_scaled = 0
            league_avg_scaled = 0
        else:
            player_scaled = ((player_raw - league_min) / (league_max - league_min)) * 100
            league_avg_scaled = ((league_avg_raw - league_min) / (league_max - league_min)) * 100

        metric_rows.append(
            {
                "Metric": metric_name,
                "Group": selected_player,
                "Score": player_scaled,
                "Raw Value": player_raw,
            }
        )

        metric_rows.append(
            {
                "Metric": metric_name,
                "Group": "League Average",
                "Score": league_avg_scaled,
                "Raw Value": league_avg_raw,
            }
        )

    player_metrics = pd.DataFrame(metric_rows)

    fig = px.bar(
        player_metrics,
        x="Metric",
        y="Score",
        color="Group",
        barmode="group",
        title=f"{selected_player} Lineup Value Breakdown vs League Average",
        hover_data=["Raw Value"],
        range_y=[0, 100],
    )

    st.plotly_chart(fig, use_container_width=True)

    shot_frequency_columns = ["freq_rim", "freq_short_paint", "freq_mid", "freq_long_mid", "freq_3p"]
    shot_accuracy_columns = ["fg_rim", "fg_short_paint", "fg_mid", "fg_long_mid", "fg_3p"]

    if all(column in df.columns for column in shot_frequency_columns):
        st.subheader("Shot Profile vs Position Average")

        shot_labels = ["Rim", "Short Paint", "Midrange", "Long Midrange", "Three"]

        player_position = player.get("pos", None)

        if "pos" in df.columns and pd.notna(player_position):
            position_df = df[df["pos"] == player_position].copy()
        else:
            position_df = df.copy()

        shot_freq_rows = []

        for zone, column in zip(shot_labels, shot_frequency_columns):
            shot_freq_rows.append(
                {
                    "Zone": zone,
                    "Group": selected_player,
                    "Frequency": player.get(column, np.nan),
                }
            )


            shot_freq_rows.append(
                {
                    "Zone": zone,
                    "Group": f"{player_position} Average",
                    "Frequency": position_df[column].mean(),
                }
            )

        shot_freq_df = pd.DataFrame(shot_freq_rows)

        fig_freq = px.bar(
            shot_freq_df,
            x="Zone",
            y="Frequency",
            color="Group",
            barmode="group",
            title=f"{selected_player} Shot Frequency vs {player_position} Average",
        )

        st.plotly_chart(fig_freq, use_container_width=True)

        if all(column in df.columns for column in shot_accuracy_columns):
            shot_acc_rows = []

            for zone, column in zip(shot_labels, shot_accuracy_columns):
                shot_acc_rows.append(
                    {
                        "Zone": zone,
                        "Group": selected_player,
                        "FG%": player.get(column, np.nan),
                    }
                )

                shot_acc_rows.append(
                    {
                        "Zone": zone,
                        "Group": f"{player_position} Average",
                        "FG%": position_df[column].mean(),
                    }
                )

            shot_acc_df = pd.DataFrame(shot_acc_rows)

            fig_fg = px.bar(
                shot_acc_df,
                x="Zone",
                y="FG%",
                color="Group",
                barmode="group",
                title=f"{selected_player} Shot Accuracy vs {player_position} Average",
            )

            st.plotly_chart(fig_fg, use_container_width=True)

    with st.expander("View full player row"):
        st.dataframe(
            pd.DataFrame([player[available_columns(df, PROFILE_COLUMNS)].to_dict()]).T.rename(columns={0: "Value"}),
            use_container_width=True,
        )


def show_lineup_builder_tab(df: pd.DataFrame) -> None:
    st.header("Lineup Builder")
    st.write("Create a suggested lineup based on the style of team you want to build.")

    build_type = st.selectbox(
        "Lineup style",
        ["Balanced", "Offense First", "Defense First", "Spacing First", "Playmaking First", "Rebounding First"],
    )

    weights_by_type = {
        "Balanced": {"Offense": 40, "Defense": 25, "Spacing": 15, "Playmaking": 10, "Rebounding": 10},
        "Offense First": {"Offense": 55, "Defense": 15, "Spacing": 15, "Playmaking": 10, "Rebounding": 5},
        "Defense First": {"Offense": 20, "Defense": 50, "Spacing": 10, "Playmaking": 5, "Rebounding": 15},
        "Spacing First": {"Offense": 25, "Defense": 15, "Spacing": 45, "Playmaking": 10, "Rebounding": 5},
        "Playmaking First": {"Offense": 25, "Defense": 15, "Spacing": 15, "Playmaking": 35, "Rebounding": 10},
        "Rebounding First": {"Offense": 20, "Defense": 20, "Spacing": 10, "Playmaking": 5, "Rebounding": 45},
    }

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        min_games = st.slider("Minimum games", 0, int(df["games_played"].max()), 20, key="lineup_min_games")
    with filter_col2:
        min_minutes = st.slider("Minimum average minutes", 0, int(df["avg_minutes"].max()), 10)
    with filter_col3:
        number_of_players = st.slider("Lineup size", 5, 10, 5)

    pool = df[(df["games_played"] >= min_games) & (df["avg_minutes"] >= min_minutes)].copy()
    pool["custom_lineup_score"] = build_custom_lineup_score(pool, weights_by_type[build_type])
    suggested = pool.sort_values("custom_lineup_score", ascending=False).head(number_of_players)

    st.subheader(f"Suggested {build_type} Lineup")
    card_columns = st.columns(min(number_of_players, 5))
    for index, (_, player) in enumerate(suggested.head(5).reset_index(drop=True).iterrows()):
        with card_columns[index]:
            show_player_card(player)

    st.dataframe(
        suggested[
            available_columns(
                suggested,
                [
                    "playerName",
                    "team",
                    "pos",
                    "custom_lineup_score",
                    "lineup_score",
                    "impact_score",
                    "offensive_creation",
                    "lineup_defensive_impact",
                    "spacing_value",
                    "playmaking_value",
                    "rebounding_value",
                    "lineup_role",
                ],
            )
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )

    lineup_scores = calculate_lineup_score(suggested.assign(rotation_role="Starter"), df)
    score_df = pd.DataFrame(
        {
            "Category": ["Offense", "Defense", "Spacing", "Playmaking", "Rebounding"],
            "Score": [
                lineup_scores["offense_score"],
                lineup_scores["defense_score"],
                lineup_scores["spacing_score"],
                lineup_scores["playmaking_score"],
                lineup_scores["rebounding_score"],
            ],
        }
    )
    fig = px.bar(score_df, x="Category", y="Score", title="Suggested Lineup Strengths", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def show_matchup_simulator_tab(df: pd.DataFrame) -> None:
    st.header("Matchup Simulator")
    st.write("Build two custom rotations, compare lineup strengths, and generate a projected final score and box score.")

    player_list = sorted(df["playerName"].dropna().unique())

    team_a_column, team_b_column = st.columns(2)
    with team_a_column:
        team_a_starters, team_a_bench = select_team("Team A", player_list)
    with team_b_column:
        team_b_starters, team_b_bench = select_team("Team B", player_list)

    show_duplicate_player_warning(team_a_starters, team_a_bench, team_b_starters, team_b_bench)

    if len(team_a_starters) != 5 or len(team_b_starters) != 5:
        st.info("Choose exactly 5 starters for both teams. Bench players are optional.")
        return

    team_a_rotation = build_rotation(df, team_a_starters, team_a_bench)
    team_b_rotation = build_rotation(df, team_b_starters, team_b_bench)

    team_a_scores = calculate_lineup_score(team_a_rotation, df)
    team_b_scores = calculate_lineup_score(team_b_rotation, df)
    team_a_points, team_b_points = project_final_score(team_a_scores, team_b_scores)

    st.subheader("Projected Result")
    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric("Team A", team_a_points)
    score_col2.metric("Team B", team_b_points)

    if team_a_points > team_b_points:
        score_col3.success("Projected Winner: Team A")
    elif team_b_points > team_a_points:
        score_col3.success("Projected Winner: Team B")
    else:
        score_col3.warning("Projected Result: Tie")

    st.markdown(f"## Projected Final: Team A {team_a_points} - Team B {team_b_points}")

    rotation_col1, rotation_col2 = st.columns(2)
    with rotation_col1:
        show_rotation(team_a_rotation, "Team A Rotation")
    with rotation_col2:
        show_rotation(team_b_rotation, "Team B Rotation")

    team_a_box = create_projected_box_score(team_a_rotation, team_a_points)
    team_b_box = create_projected_box_score(team_b_rotation, team_b_points)

    st.subheader("Projected Team Box Score")
    box_col1, box_col2 = st.columns(2)
    with box_col1:
        st.markdown("### Team A")
        st.dataframe(create_team_box_score(team_a_box), use_container_width=True)
    with box_col2:
        st.markdown("### Team B")
        st.dataframe(create_team_box_score(team_b_box), use_container_width=True)

    st.subheader("Projected Player Box Score")
    player_box_col1, player_box_col2 = st.columns(2)
    with player_box_col1:
        st.markdown("### Team A")
        st.dataframe(
            team_a_box.sort_values(["rotation_role", "PTS"], ascending=[False, False]),
            use_container_width=True,
            hide_index=True,
        )
    with player_box_col2:
        st.markdown("### Team B")
        st.dataframe(
            team_b_box.sort_values(["rotation_role", "PTS"], ascending=[False, False]),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Matchup Breakdown")
    show_matchup_breakdown(team_a_scores, team_b_scores)


def show_visualizations_tab(df: pd.DataFrame) -> None:
    st.header("Visualizations")
    st.write("Explore relationships between player impact, usage, efficiency, role, and lineup value.")

    numeric_options = [
        "impact_score",
        "lineup_score",
        "offensive_creation",
        "lineup_defensive_impact",
        "spacing_value",
        "playmaking_value",
        "rebounding_value",
        "ts_pct",
        "usg_pct",
        "pie",
        "net_rating",
        "avg_minutes",
        "games_played",
    ]
    numeric_options = available_columns(df, numeric_options)

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        x_axis = st.selectbox("X-axis", numeric_options, index=numeric_options.index("usg_pct"))
    with chart_col2:
        y_axis = st.selectbox("Y-axis", numeric_options, index=numeric_options.index("ts_pct"))
    with chart_col3:
        size_metric = st.selectbox("Bubble size", numeric_options, index=numeric_options.index("impact_score"))

    scatter_df = df.copy()
    scatter_df["bubble_size"] = make_positive_marker_size(scatter_df, size_metric)

    fig = px.scatter(
        scatter_df,
        x=x_axis,
        y=y_axis,
        size="bubble_size",
        color="lineup_role" if "lineup_role" in scatter_df.columns else None,
        hover_name="playerName",
        hover_data=available_columns(scatter_df, ["team", "pos", "impact_score", "lineup_score", "ts_pct", "usg_pct"]),
        title=f"{y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig, use_container_width=True)

    hist_col1, hist_col2 = st.columns(2)
    with hist_col1:
        fig_hist = px.histogram(df, x="impact_score", nbins=30, title="Impact Score Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
    with hist_col2:
        role_counts = df["lineup_role"].value_counts().reset_index()
        role_counts.columns = ["Lineup Role", "Players"]
        fig_role = px.bar(role_counts, x="Lineup Role", y="Players", title="Player Count by Lineup Role")
        st.plotly_chart(fig_role, use_container_width=True)


def show_data_explorer_tab(df: pd.DataFrame) -> None:
    st.header("Data Explorer")
    st.write("Filter and inspect the final player profile dataset used by the app.")

    filtered = df.copy()

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        if "team" in df.columns:
            teams = sorted(df["team"].dropna().unique())
            selected_teams = st.multiselect("Team", teams)
            if selected_teams:
                filtered = filtered[filtered["team"].isin(selected_teams)]

    with filter_col2:
        if "pos" in df.columns:
            positions = sorted(df["pos"].dropna().unique())
            selected_positions = st.multiselect("Position", positions)
            if selected_positions:
                filtered = filtered[filtered["pos"].isin(selected_positions)]

    with filter_col3:
        if "lineup_role" in df.columns:
            roles = sorted(df["lineup_role"].dropna().unique())
            selected_roles = st.multiselect("Lineup Role", roles)
            if selected_roles:
                filtered = filtered[filtered["lineup_role"].isin(selected_roles)]

    search_text = st.text_input("Search player name")
    if search_text:
        filtered = filtered[filtered["playerName"].str.contains(search_text, case=False, na=False)]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} players")
    st.dataframe(filtered.round(3), use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        csv,
        "filtered_nba_player_profiles.csv",
        "text/csv",
    )


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main() -> None:
    st.title("🏀 NBA Player Impact Analyzer")
    st.caption("Player rankings, hidden gems, profile scouting, lineup building, and matchup simulation.")

    try:
        players_df = load_data()
    except Exception as error:
        st.error(f"Could not load player data: {error}")
        st.stop()

    with st.sidebar:
        st.header("Dashboard Controls")
        st.metric("Players loaded", len(players_df))
        st.caption("Data source: final processed NBA player profile dataset")
        st.markdown("---")
        st.write("Use the tabs across the top to move through each feature.")
        st.info("Bench players are optional in the matchup simulator. Starters are required for both teams.")

    overview_tab, rankings_tab, hidden_gems_tab, profiles_tab, lineup_tab, matchup_tab, visuals_tab, data_tab = st.tabs(
        [
            "🏠 Overview",
            "⭐ Rankings",
            "💎 Hidden Gems",
            "👤 Player Profiles",
            "🧩 Lineup Builder",
            "⚔️ Matchup Simulator",
            "📊 Visualizations",
            "🗂️ Data Explorer",
        ]
    )

    with overview_tab:
        show_overview_tab(players_df)

    with rankings_tab:
        show_rankings_tab(players_df)

    with hidden_gems_tab:
        show_hidden_gems_tab(players_df)

    with profiles_tab:
        show_player_profiles_tab(players_df)

    with lineup_tab:
        show_lineup_builder_tab(players_df)

    with matchup_tab:
        show_matchup_simulator_tab(players_df)

    with visuals_tab:
        show_visualizations_tab(players_df)

    with data_tab:
        show_data_explorer_tab(players_df)


if __name__ == "__main__":
    main()
