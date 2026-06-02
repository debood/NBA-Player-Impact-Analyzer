from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NBA Player Impact Analyzer",
    page_icon="🏀",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "player_profiles_final.csv"
NBA_HEADSHOT_BASE_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760"
REQUIRED_COLUMNS = [
    "playerName",
    "personId",
    "lineup_role",
    "lineup_score",
    "offensive_creation",
    "lineup_defensive_impact",
    "spacing_value",
    "playmaking_value",
    "rebounding_value",
    "points_generated_by_assists",
    "pct_reb",
    "ts_pct",
    "scorer_profile",
    "defensive_profile",
]


@st.cache_data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load final player profiles and create NBA CDN headshot URLs from personId."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["playerName"]).copy()
    df["playerName"] = df["playerName"].astype(str).str.strip()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["personId"] = pd.to_numeric(df["personId"], errors="coerce").astype("Int64")
    df["headshot_url"] = df["personId"].apply(
        lambda player_id: f"{NBA_HEADSHOT_BASE_URL}/{int(player_id)}.png" if pd.notna(player_id) else None
    )

    return df.sort_values("playerName").reset_index(drop=True)


def min_max_scale_value(value: float, series: pd.Series) -> float:
    clean_series = pd.to_numeric(series, errors="coerce").fillna(0)
    clean_value = 0 if pd.isna(value) else value

    if clean_series.max() == clean_series.min():
        return 0.0

    return float((clean_value - clean_series.min()) / (clean_series.max() - clean_series.min()))


def calculate_lineup_score(lineup: pd.DataFrame, all_players: pd.DataFrame) -> dict:
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
    base_score = 112
    score_gap = team_a_scores["lineup_score"] - team_b_scores["lineup_score"]
    projected_margin = score_gap * 28

    team_a_points = base_score + projected_margin / 2
    team_b_points = base_score - projected_margin / 2

    team_a_points = max(85, min(140, team_a_points))
    team_b_points = max(85, min(140, team_b_points))

    return round(team_a_points), round(team_b_points)


def create_projected_box_score(rotation: pd.DataFrame, target_team_points: int) -> pd.DataFrame:
    box = rotation.copy()
    box["MIN"] = np.where(box["rotation_role"] == "Starter", 34, 14)

    total_minutes = box["MIN"].sum()
    if total_minutes > 0:
        box["MIN"] = box["MIN"] * (240 / total_minutes)

    box["point_weight"] = (
        box["offensive_creation"].fillna(0) * 0.70
        + box["lineup_score"].fillna(0) * 30
    )
    if box["point_weight"].sum() == 0:
        box["point_weight"] = 1

    box["PTS"] = (box["point_weight"] / box["point_weight"].sum()) * target_team_points

    box["AST"] = box["points_generated_by_assists"].fillna(0)
    box["AST"] = (box["AST"] / box["AST"].sum()) * 25 if box["AST"].sum() > 0 else 0

    box["REB"] = box["pct_reb"].fillna(0)
    box["REB"] = (box["REB"] / box["REB"].sum()) * 44 if box["REB"].sum() > 0 else 0

    box["STL"] = box.get("pct_stl", pd.Series(0, index=box.index)).fillna(0)
    box["BLK"] = box.get("pct_blk", pd.Series(0, index=box.index)).fillna(0)

    if box["STL"].sum() > 0:
        box["STL"] = (box["STL"] / box["STL"].sum()) * 7
    if box["BLK"].sum() > 0:
        box["BLK"] = (box["BLK"] / box["BLK"].sum()) * 5

    box["FGM"] = box["PTS"] / 2.25
    box["FGA"] = box["FGM"] / box["ts_pct"].replace(0, np.nan).fillna(0.55)

    display_columns = [
        "rotation_role",
        "playerName",
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

    for column in ["MIN", "PTS", "AST", "REB", "STL", "BLK", "FGM", "FGA"]:
        box[column] = box[column].round(1)

    return box[display_columns]


def create_team_box_score(player_box: pd.DataFrame) -> pd.DataFrame:
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

    team_box["FG%"] = team_box["FGM"] / team_box["FGA"].replace(0, np.nan)
    return team_box.round(3)


def show_player_card(player: pd.Series) -> None:
    if pd.notna(player.get("headshot_url")):
        st.image(player["headshot_url"], width=105)

    st.markdown(f"**{player['playerName']}**")
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
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("🏀 NBA Player Impact Analyzer")
    st.write(
        "Build two custom NBA rotations, compare lineup strengths, and generate a projected final score and box score."
    )

    try:
        players_df = load_data()
    except Exception as error:
        st.error(f"Could not load player data: {error}")
        st.stop()

    with st.sidebar:
        st.header("About this app")
        st.write(
            "Scores are based on player profile metrics for offense, defense, spacing, playmaking, and rebounding."
        )
        st.metric("Players loaded", len(players_df))
        st.caption("Bench players are optional. Starters are required for both teams.")

    player_list = sorted(players_df["playerName"].dropna().unique())

    team_a_column, team_b_column = st.columns(2)
    with team_a_column:
        team_a_starters, team_a_bench = select_team("Team A", player_list)
    with team_b_column:
        team_b_starters, team_b_bench = select_team("Team B", player_list)

    show_duplicate_player_warning(team_a_starters, team_a_bench, team_b_starters, team_b_bench)

    if len(team_a_starters) != 5 or len(team_b_starters) != 5:
        st.info("Choose exactly 5 starters for both teams. Bench players are optional.")
        return

    team_a_rotation = build_rotation(players_df, team_a_starters, team_a_bench)
    team_b_rotation = build_rotation(players_df, team_b_starters, team_b_bench)

    team_a_scores = calculate_lineup_score(team_a_rotation, players_df)
    team_b_scores = calculate_lineup_score(team_b_rotation, players_df)
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

    show_rotation(team_a_rotation, "Team A Rotation")
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
        )
    with player_box_col2:
        st.markdown("### Team B")
        st.dataframe(
            team_b_box.sort_values(["rotation_role", "PTS"], ascending=[False, False]),
            use_container_width=True,
        )

    st.subheader("Matchup Breakdown")
    show_matchup_breakdown(team_a_scores, team_b_scores)


if __name__ == "__main__":
    main()
