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
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_cricket_model_14f.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "model_features.pkl"
)

ENRICHED_PATH = os.path.join(
    BASE_DIR,
    "df_enriched.csv"
)

MATCH_PLAYERS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "match_players.csv"
)

SUBSTITUTES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "substitutes.csv"
)


# ============================================================
# LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_artifacts():

    # --------------------------------------------------------
    # ML MODEL
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    features = joblib.load(
        FEATURES_PATH
    )

    # --------------------------------------------------------
    # ENRICHED DATASET
    # --------------------------------------------------------

    df_enriched = pd.read_csv(
        ENRICHED_PATH
    )

    # --------------------------------------------------------
    # MATCH PLAYER DATA
    # --------------------------------------------------------

    match_players = pd.read_csv(
        MATCH_PLAYERS_PATH
    )

    # --------------------------------------------------------
    # SUBSTITUTE DATA
    # --------------------------------------------------------

    substitutes = pd.read_csv(
        SUBSTITUTES_PATH
    )

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

    # --------------------------------------------------------
    # SORT DATA
    # --------------------------------------------------------

    df_enriched = df_enriched.sort_values(
        "date"
    ).reset_index(drop=True)

    match_players = match_players.sort_values(
        ["date", "team", "player"]
    ).reset_index(drop=True)

    substitutes = substitutes.sort_values(
        ["date", "substitute_player"]
    ).reset_index(drop=True)

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
# VERIFY IMPORTANT COLUMNS
# ============================================================

required_player_columns = [
    "match_id",
    "date",
    "team",
    "player",
    "status"
]

missing_player_columns = [
    col
    for col in required_player_columns
    if col not in match_players.columns
]

if missing_player_columns:

    st.error(
        "match_players.csv is missing these columns: "
        + ", ".join(missing_player_columns)
    )

    st.stop()


required_substitute_columns = [
    "match_id",
    "date",
    "team_1",
    "team_2",
    "substitute_player",
    "event"
]

missing_substitute_columns = [
    col
    for col in required_substitute_columns
    if col not in substitutes.columns
]

if missing_substitute_columns:

    st.error(
        "substitutes.csv is missing these columns: "
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
        .union(
            set(
                df_enriched["team_2"]
                .dropna()
                .unique()
            )
        )
    )
)


# ============================================================
# VENUE LIST
# ============================================================

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
        min_value=available_dates[0],
        max_value=available_dates[-1],
        format="DD/MM/YYYY"
    )

else:

    selected_date = None

    st.error(
        "No valid match dates found in df_enriched.csv."
    )

    st.stop()


# ============================================================
# TEAM SELECTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    team1 = st.selectbox(
        "Select Team 1",
        options=all_teams,
        index=0,
        key="team1"
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
        index=0,
        key="team2"
    )


# ============================================================
# FIND PLAYER DATA FOR SELECTED DATE + TEAMS
# ============================================================

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

st.markdown("---")

st.header("👥 Players for Selected Match")


if selected_match_players.empty:

    st.warning(
        "⚠️ No player information was found "
        "for these teams on the selected date."
    )

    st.info(
        "Try selecting the actual IPL match date "
        "for which player data is available."
    )

else:

    # --------------------------------------------------------
    # TEAM 1
    # --------------------------------------------------------

    team1_players = selected_match_players[
        selected_match_players["team"]
        == team1
    ].copy()

    team1_official = team1_players[
        team1_players["status"]
        == "officially_listed"
    ].copy()

    team1_substitutes = team1_players[
        team1_players["status"]
        != "officially_listed"
    ].copy()


    st.subheader(
        f"🏏 {team1}"
    )


    if not team1_official.empty:

        st.markdown(
            "**👥 Officially Listed Players:**"
        )

        for i, player in enumerate(
            team1_official["player"].tolist(),
            start=1
        ):

            st.write(
                f"{i}. {player}"
            )

    else:

        st.info(
            "No officially listed players found."
        )


    if not team1_substitutes.empty:

        st.markdown(
            "**🔄 Substitute Fielders Who Appeared:**"
        )

        for player in team1_substitutes[
            "player"
        ].tolist():

            st.write(
                f"• {player}"
            )


    # --------------------------------------------------------
    # TEAM 2
    # --------------------------------------------------------

    team2_players = selected_match_players[
        selected_match_players["team"]
        == team2
    ].copy()

    team2_official = team2_players[
        team2_players["status"]
        == "officially_listed"
    ].copy()

    team2_substitutes = team2_players[
        team2_players["status"]
        != "officially_listed"
    ].copy()


    st.subheader(
        f"🏏 {team2}"
    )


    if not team2_official.empty:

        st.markdown(
            "**👥 Officially Listed Players:**"
        )

        for i, player in enumerate(
            team2_official["player"].tolist(),
            start=1
        ):

            st.write(
                f"{i}. {player}"
            )

    else:

        st.info(
            "No officially listed players found."
        )


    if not team2_substitutes.empty:

        st.markdown(
            "**🔄 Substitute Fielders Who Appeared:**"
        )

        for player in team2_substitutes[
            "player"
        ].tolist():

            st.write(
                f"• {player}"
            )


# ============================================================
# SUBSTITUTE EVENTS
# ============================================================

match_substitutes = substitutes[
    (
        substitutes["date"].dt.date
        == selected_date
    )
    &
    (
        (
            (
                substitutes["team_1"]
                == team1
            )
            &
            (
                substitutes["team_2"]
                == team2
            )
        )
        |
        (
            (
                substitutes["team_1"]
                == team2
            )
            &
            (
                substitutes["team_2"]
                == team1
            )
        )
    )
].copy()


if not match_substitutes.empty:

    st.markdown("---")

    st.header("🔄 Substitute Events")


    displayed_substitutes = (
        match_substitutes[
            [
                "substitute_player",
                "event"
            ]
        ]
        .drop_duplicates()
    )


    for _, row in displayed_substitutes.iterrows():

        st.write(
            f"• **{row['substitute_player']}** "
            f"— {row['event']}"
        )


# ============================================================
# MATCH CONDITIONS
# ============================================================

st.markdown("---")

st.header("🏟️ Match Conditions")


venue = st.selectbox(
    "Select Venue",
    options=all_venues,
    key="venue"
)


# ============================================================
# TOSS
# ============================================================

col3, col4 = st.columns(2)


with col3:

    toss_winner = st.radio(
        "🪙 Toss Winner",
        options=[
            team1,
            team2
        ],
        key="toss_winner"
    )


with col4:

    batted_first = st.radio(
        "🏏 Team Batting First",
        options=[
            team1,
            team2
        ],
        key="batted_first"
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_match_ui(
    t1,
    t2,
    toss_w,
    bat_1st,
    ven,
    prediction_date
):

    # ========================================================
    # GET TEAM STATISTICS
    # ========================================================

    def get_latest_stats(
        team,
        venue_name,
        prediction_date
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        # Only use matches BEFORE prediction date.
        # This prevents data leakage.
        # ----------------------------------------------------

        team_matches = df_enriched[
            (
                (
                    df_enriched["team_1"]
                    == team
                )
                |
                (
                    df_enriched["team_2"]
                    == team
                )
            )
            &
            (
                df_enriched["date"]
                < pd.Timestamp(
                    prediction_date
                )
            )
        ].copy()


        if team_matches.empty:

            return {
                "win_rate": 0.5,
                "recent_win_rate": 0.5,
                "venue_win_rate": 0.5
            }


        # ----------------------------------------------------
        # SORT CHRONOLOGICALLY
        # ----------------------------------------------------

        team_matches = team_matches.sort_values(
            "date"
        ).reset_index(
            drop=True
        )


        # ----------------------------------------------------
        # LATEST MATCH
        # ----------------------------------------------------

        last_match = team_matches.iloc[-1]


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
        # VENUE STATISTICS
        # ----------------------------------------------------

        venue_matches = team_matches[
            team_matches["venue"]
            == venue_name
        ].copy()


        if not venue_matches.empty:

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
            "win_rate": float(wr),
            "recent_win_rate": float(rwr),
            "venue_win_rate": float(vwr)
        }


    # ========================================================
    # TEAM 1 STATS
    # ========================================================

    t1_s = get_latest_stats(
        t1,
        ven,
        prediction_date
    )


    # ========================================================
    # TEAM 2 STATS
    # ========================================================

    t2_s = get_latest_stats(
        t2,
        ven,
        prediction_date
    )


    # ========================================================
    # HEAD-TO-HEAD
    # ========================================================

    h2h = df_enriched[
        (
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
        )
        &
        (
            df_enriched["date"]
            < pd.Timestamp(
                prediction_date
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


    # Make sure feature order is EXACTLY
    # the same as training.

    input_df = input_df[
        features
    ]


    # ========================================================
    # PREDICTION
    # ========================================================

    probabilities = model.predict_proba(
        input_df
    )[0]


    return probabilities


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")


if st.button(
    "🚀 Predict Match Outcome",
    use_container_width=True
):

    try:

        probs = predict_match_ui(
            team1,
            team2,
            toss_winner,
            batted_first,
            venue,
            selected_date
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        t1_prob = round(
            probs[0] * 100,
            2
        )

        t2_prob = round(
            probs[1] * 100,
            2
        )


        # ====================================================
        # WINNER
        # ====================================================

        if t1_prob > t2_prob:

            winner = team1

        else:

            winner = team2


        # ====================================================
        # CONFIDENCE
        # ====================================================

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


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.header(
            "🎯 Prediction Result"
        )


        st.success(
            f"🏆 Predicted Winner: **{winner}**"
        )


        # ====================================================
        # PROBABILITY COLUMNS
        # ====================================================

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


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📊 Prediction Confidence"
        )


        st.metric(
            "Confidence Level",
            confidence_level
        )


        st.write(
            f"Confidence Percentage: "
            f"**{confidence}%**"
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🤖 Model Information"
        )


        st.write(
            "**Model:** "
            "14-Feature Logistic Regression"
        )

        st.write(
            f"**Number of Features:** "
            f"{len(features)}"
        )


        # ====================================================
        # MATCH INFORMATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🏏 Match Information"
        )


        st.write(
            f"**Date:** "
            f"{selected_date.strftime('%d/%m/%Y')}"
        )

        st.write(
            f"**Team 1:** "
            f"{team1}"
        )

        st.write(
            f"**Team 2:** "
            f"{team2}"
        )

        st.write(
            f"**Venue:** "
            f"{venue}"
        )

        st.write(
            f"**Toss Winner:** "
            f"{toss_winner}"
        )

        st.write(
            f"**Batting First:** "
            f"{batted_first}"
        )


        # ====================================================
        # PLAYER SUMMARY IN RESULT
        # ====================================================

        st.markdown("---")

        st.subheader(
            "👥 Selected Match Players"
        )


        if selected_match_players.empty:

            st.info(
                "No player information is available "
                "for this date and team combination."
            )

        else:

            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.markdown(
                    f"### 🏏 {team1}"
                )


                result_t1 = selected_match_players[
                    selected_match_players["team"]
                    == team1
                ]


                result_t1_official = result_t1[
                    result_t1["status"]
                    == "officially_listed"
                ]


                for i, player in enumerate(
                    result_t1_official[
                        "player"
                    ].tolist(),
                    start=1
                ):

                    st.write(
                        f"{i}. {player}"
                    )


                result_t1_subs = result_t1[
                    result_t1["status"]
                    != "officially_listed"
                ]


                if not result_t1_subs.empty:

                    st.markdown(
                        "**🔄 Substitutes:**"
                    )

                    for player in result_t1_subs[
                        "player"
                    ].tolist():

                        st.write(
                            f"• {player}"
                        )


            with result_col2:

                st.markdown(
                    f"### 🏏 {team2}"
                )


                result_t2 = selected_match_players[
                    selected_match_players["team"]
                    == team2
                ]


                result_t2_official = result_t2[
                    result_t2["status"]
                    == "officially_listed"
                ]


                for i, player in enumerate(
                    result_t2_official[
                        "player"
                    ].tolist(),
                    start=1
                ):

                    st.write(
                        f"{i}. {player}"
                    )


                result_t2_subs = result_t2[
                    result_t2["status"]
                    != "officially_listed"
                ]


                if not result_t2_subs.empty:

                    st.markdown(
                        "**🔄 Substitutes:**"
                    )

                    for player in result_t2_subs[
                        "player"
                    ].tolist():

                        st.write(
                            f"• {player}"
                        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)