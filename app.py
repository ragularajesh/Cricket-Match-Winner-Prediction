import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cricket Winner Predictor",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 Cricket Match Winner Predictor")

st.markdown(
    "Predict match win probabilities using the "
    "**14-Feature Logistic Regression Model**."
)

# ============================================================
# LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_artifacts():

    # --------------------------------------------------------
    # ML MODEL
    # --------------------------------------------------------

    model = joblib.load("best_cricket_model_14f.pkl")

    # Features used during training
    features = joblib.load("model_features.pkl")

    # Main enriched dataset
    df_enriched = pd.read_csv("df_enriched.csv")

    # --------------------------------------------------------
    # PLAYER DATASET
    # --------------------------------------------------------

    players_path = os.path.join(
        "data",
        "cleaned",
        "match_players.csv"
    )

    # --------------------------------------------------------
    # SUBSTITUTE DATASET
    # --------------------------------------------------------

    substitutes_path = os.path.join(
        "data",
        "cleaned",
        "substitutes.csv"
    )

    match_players = pd.read_csv(players_path)

    substitutes = pd.read_csv(substitutes_path)

    # --------------------------------------------------------
    # CONVERT DATES
    # --------------------------------------------------------

    df_enriched["date"] = pd.to_datetime(
        df_enriched["date"],
        errors="coerce"
    )

    match_players["date"] = pd.to_datetime(
        match_players["date"],
        errors="coerce"
    )

    substitutes["date"] = pd.to_datetime(
        substitutes["date"],
        errors="coerce"
    )

    return (
        model,
        features,
        df_enriched,
        match_players,
        substitutes
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        model,
        features,
        df_enriched,
        match_players,
        substitutes
    ) = load_artifacts()

except Exception as e:

    st.error(
        f"""
        Error loading project files:

        {e}

        Make sure these files exist:

        best_cricket_model_14f.pkl
        model_features.pkl
        df_enriched.csv
        data/cleaned/match_players.csv
        data/cleaned/substitutes.csv
        """
    )

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_player_columns = [
    "match_id",
    "date",
    "team",
    "player",
    "status"
]

required_substitute_columns = [
    "match_id",
    "date",
    "team_1",
    "team_2",
    "substitute_player"
]

missing_player_columns = [
    col
    for col in required_player_columns
    if col not in match_players.columns
]

missing_substitute_columns = [
    col
    for col in required_substitute_columns
    if col not in substitutes.columns
]

if missing_player_columns:

    st.error(
        "match_players.csv is missing columns: "
        + ", ".join(missing_player_columns)
    )

    st.stop()

if missing_substitute_columns:

    st.error(
        "substitutes.csv is missing columns: "
        + ", ".join(missing_substitute_columns)
    )

    st.stop()


# ============================================================
# TEAM LIST
# ============================================================

all_teams = sorted(
    list(
        set(
            df_enriched["team_1"]
            .dropna()
            .unique()
        )
        |
        set(
            df_enriched["team_2"]
            .dropna()
            .unique()
        )
    )
)

all_venues = sorted(
    df_enriched["venue"]
    .dropna()
    .unique()
    .tolist()
)


# ============================================================
# MATCH SETUP
# ============================================================

st.header("📋 Match Setup")


# ============================================================
# MATCH DATE
# ============================================================

available_dates = sorted(
    df_enriched["date"]
    .dropna()
    .dt.date
    .unique()
)

if available_dates:

    selected_date = st.date_input(
        "📅 Match Date",
        value=available_dates[-1],
        min_value=min(available_dates),
        max_value=max(available_dates),
        format="DD/MM/YYYY"
    )

else:

    selected_date = None


# ============================================================
# TEAMS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    team1 = st.selectbox(
        "Select Team 1",
        options=all_teams,
        index=0
    )

remaining_teams = [
    team
    for team in all_teams
    if team != team1
]

with col2:

    team2 = st.selectbox(
        "Select Team 2",
        options=remaining_teams,
        index=0
    )


# ============================================================
# FIND PLAYERS FOR SELECTED DATE + TEAMS
# ============================================================

selected_match_players = pd.DataFrame()

if selected_date is not None:

    selected_match_players = match_players[
        (
            match_players["date"].dt.date
            == selected_date
        )
        &
        (
            match_players["team"].isin(
                [team1, team2]
            )
        )
    ].copy()


# ============================================================
# DISPLAY PLAYER INFORMATION
# ============================================================

if not selected_match_players.empty:

    st.markdown("---")

    st.header("👥 Players for Selected Match")

    # ========================================================
    # TEAM 1
    # ========================================================

    team1_players = selected_match_players[
        selected_match_players["team"] == team1
    ].copy()

    team1_official = team1_players[
        team1_players["status"]
        == "officially_listed"
    ]

    team1_substitutes = team1_players[
        team1_players["status"]
        != "officially_listed"
    ]

    st.subheader(f"🏏 {team1}")

    if not team1_official.empty:

        st.markdown(
            "**Playing XI / Officially Listed Players:**"
        )

        for i, player in enumerate(
            team1_official["player"].tolist(),
            start=1
        ):

            st.write(
                f"{i}. {player}"
            )

    if not team1_substitutes.empty:

        st.markdown(
            "**🔄 Substitute Players:**"
        )

        for player in team1_substitutes[
            "player"
        ].tolist():

            st.write(
                f"• {player}"
            )

    # ========================================================
    # TEAM 2
    # ========================================================

    team2_players = selected_match_players[
        selected_match_players["team"] == team2
    ].copy()

    team2_official = team2_players[
        team2_players["status"]
        == "officially_listed"
    ]

    team2_substitutes = team2_players[
        team2_players["status"]
        != "officially_listed"
    ]

    st.subheader(f"🏏 {team2}")

    if not team2_official.empty:

        st.markdown(
            "**Playing XI / Officially Listed Players:**"
        )

        for i, player in enumerate(
            team2_official["player"].tolist(),
            start=1
        ):

            st.write(
                f"{i}. {player}"
            )

    if not team2_substitutes.empty:

        st.markdown(
            "**🔄 Substitute Players:**"
        )

        for player in team2_substitutes[
            "player"
        ].tolist():

            st.write(
                f"• {player}"
            )


# ============================================================
# SUBSTITUTE FIELDER INFORMATION
# ============================================================

match_substitutes = pd.DataFrame()

if selected_date is not None:

    match_substitutes = substitutes[
        (
            substitutes["date"].dt.date
            == selected_date
        )
        &
        (
            (
                substitutes["team_1"].isin(
                    [team1, team2]
                )
            )
            |
            (
                substitutes["team_2"].isin(
                    [team1, team2]
                )
            )
        )
    ].copy()


if not match_substitutes.empty:

    st.markdown("---")

    st.header("🔄 Substitute Fielders")

    displayed_substitutes = set()

    for _, row in match_substitutes.iterrows():

        player = row["substitute_player"]

        if pd.isna(player):
            continue

        player = str(player)

        if player in displayed_substitutes:
            continue

        displayed_substitutes.add(player)

        st.write(
            f"• **{player}**"
        )


# ============================================================
# NO PLAYER DATA MESSAGE
# ============================================================

if selected_match_players.empty:

    st.info(
        "ℹ️ No player information was found "
        "for this date and team combination."
    )


# ============================================================
# VENUE
# ============================================================

st.markdown("---")

st.header("🏟️ Match Conditions")

venue = st.selectbox(
    "Select Venue",
    options=all_venues
)


# ============================================================
# TOSS
# ============================================================

col3, col4 = st.columns(2)

with col3:

    toss_winner = st.radio(
        "🪙 Toss Winner",
        options=[team1, team2]
    )

with col4:

    batted_first = st.radio(
        "🏏 Team Batting First",
        options=[team1, team2]
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_match_ui(
    t1,
    t2,
    toss_w,
    bat_1st,
    ven
):

    # ========================================================
    # GET LATEST TEAM STATISTICS
    # ========================================================

    def get_latest_stats(
        team,
        venue_name
    ):

        team_matches = df_enriched[
            (
                df_enriched["team_1"]
                == team
            )
            |
            (
                df_enriched["team_2"]
                == team
            )
        ].copy()

        if team_matches.empty:

            return {
                "win_rate": 0.5,
                "recent_win_rate": 0.5,
                "venue_win_rate": 0.5
            }

        team_matches = team_matches.sort_values(
            "date"
        )

        last_match = team_matches.iloc[-1]

        # ----------------------------------------------------
        # OVERALL WIN RATE
        # ----------------------------------------------------

        if last_match["team_1"] == team:

            wr = last_match[
                "team_1_win_rate"
            ]

            rwr = last_match[
                "team_1_recent_win_rate"
            ]

        else:

            wr = last_match[
                "team_2_win_rate"
            ]

            rwr = last_match[
                "team_2_recent_win_rate"
            ]

        # ----------------------------------------------------
        # VENUE WIN RATE
        # ----------------------------------------------------

        venue_matches = team_matches[
            team_matches["venue"]
            == venue_name
        ]

        if not venue_matches.empty:

            venue_matches = venue_matches.sort_values(
                "date"
            )

            v_last = venue_matches.iloc[-1]

            if v_last["team_1"] == team:

                vwr = v_last[
                    "team_1_venue_win_rate"
                ]

            else:

                vwr = v_last[
                    "team_2_venue_win_rate"
                ]

        else:

            vwr = 0.5

        return {
            "win_rate": wr,
            "recent_win_rate": rwr,
            "venue_win_rate": vwr
        }


    # ========================================================
    # TEAM STATS
    # ========================================================

    t1_s = get_latest_stats(
        t1,
        ven
    )

    t2_s = get_latest_stats(
        t2,
        ven
    )


    # ========================================================
    # HEAD TO HEAD
    # ========================================================

    h2h = df_enriched[
        (
            (
                df_enriched["team_1"]
                == t1
            )
            &
            (
                df_enriched["team_2"]
                == t2
            )
        )
        |
        (
            (
                df_enriched["team_1"]
                == t2
            )
            &
            (
                df_enriched["team_2"]
                == t1
            )
        )
    ].copy()

    if not h2h.empty:

        h2h = h2h.sort_values(
            "date"
        )

        h2h_last = h2h.iloc[-1]

        if h2h_last["team_1"] == t1:

            h2h_1 = h2h_last[
                "team_1_h2h_win_rate"
            ]

            h2h_2 = h2h_last[
                "team_2_h2h_win_rate"
            ]

        else:

            h2h_1 = h2h_last[
                "team_2_h2h_win_rate"
            ]

            h2h_2 = h2h_last[
                "team_1_h2h_win_rate"
            ]

    else:

        h2h_1 = 0.5
        h2h_2 = 0.5


    # ========================================================
    # 14 FEATURES
    # ========================================================

    match_data = {

        "team_1_win_rate":
            t1_s["win_rate"],

        "team_2_win_rate":
            t2_s["win_rate"],

        "win_rate_difference":
            t1_s["win_rate"]
            -
            t2_s["win_rate"],

        "team_1_recent_win_rate":
            t1_s["recent_win_rate"],

        "team_2_recent_win_rate":
            t2_s["recent_win_rate"],

        "recent_form_difference":
            t1_s["recent_win_rate"]
            -
            t2_s["recent_win_rate"],

        "team_1_h2h_win_rate":
            h2h_1,

        "team_2_h2h_win_rate":
            h2h_2,

        "h2h_difference":
            h2h_1
            -
            h2h_2,

        "team_1_won_toss":
            1
            if toss_w == t1
            else 0,

        "team_1_batted_first":
            1
            if bat_1st == t1
            else 0,

        "team_1_venue_win_rate":
            t1_s["venue_win_rate"],

        "team_2_venue_win_rate":
            t2_s["venue_win_rate"],

        "venue_win_rate_difference":
            t1_s["venue_win_rate"]
            -
            t2_s["venue_win_rate"]
    }


    # ========================================================
    # MODEL INPUT
    # ========================================================

    input_df = pd.DataFrame(
        [match_data]
    )

    input_df = input_df[features]


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    return model.predict_proba(
        input_df
    )[0]


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")

if st.button(
    "🚀 Predict Match Outcome",
    use_container_width=True
):

    probs = predict_match_ui(
        team1,
        team2,
        toss_winner,
        batted_first,
        venue
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    t1_prob = round(
        probs[0] * 100,
        2
    )

    t2_prob = round(
        probs[1] * 100,
        2
    )


    # ========================================================
    # WINNER
    # ========================================================

    winner = (
        team1
        if t1_prob > t2_prob
        else team2
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = max(
        t1_prob,
        t2_prob
    )

    if confidence >= 70:

        confidence_level = "High"

    elif confidence >= 60:

        confidence_level = "Medium"

    else:

        confidence_level = "Low"


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.header("🎯 Prediction Result")

    st.success(
        f"🏆 Predicted Winner: **{winner}**"
    )


    # ========================================================
    # PROBABILITY DISPLAY
    # ========================================================

    col_a, col_b = st.columns(2)

    with col_a:

        st.metric(
            f"{team1} Win Probability",
            f"{t1_prob}%"
        )

        st.progress(
            t1_prob / 100
        )

    with col_b:

        st.metric(
            f"{team2} Win Probability",
            f"{t2_prob}%"
        )

        st.progress(
            t2_prob / 100
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Prediction Confidence"
    )

    st.metric(
        "Confidence",
        confidence_level
    )

    st.write(
        f"Confidence Percentage: "
        f"**{confidence}%**"
    )


    # ========================================================
    # MATCH INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🏏 Match Information"
    )

    st.write(
        f"**Date:** {selected_date}"
    )

    st.write(
        f"**Team 1:** {team1}"
    )

    st.write(
        f"**Team 2:** {team2}"
    )

    st.write(
        f"**Venue:** {venue}"
    )

    st.write(
        f"**Toss Winner:** {toss_winner}"
    )

    st.write(
        f"**Batting First:** {batted_first}"
    )
