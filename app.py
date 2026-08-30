import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cricket Match Winner Prediction",
    page_icon="🏏",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cricket_winner_random_forest.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "combined_features.csv"
)

PLAYERS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "match_players.csv"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


@st.cache_data
def load_players():
    if os.path.exists(PLAYERS_PATH):
        data = pd.read_csv(PLAYERS_PATH)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        return data

    return pd.DataFrame()


model = load_model()
df = load_data()
players = load_players()

st.title("🏏 Cricket Match Winner Prediction")
st.write("IPL • T20I • ODI • Test")

st.markdown("---")

st.header("🏏 Match Format")

selected_format = st.selectbox(
    "Select Match Type",
    ["IPL", "T20I", "ODI", "Test"]
)

format_df = df[
    df["match_type"].astype(str).str.strip().str.lower()
    == selected_format.lower()
].copy()

if format_df.empty:
    st.error("No matches found for " + selected_format)
    st.stop()

st.success(
    selected_format
    + ": "
    + str(len(format_df))
    + " matches available"
)


teams = sorted(
    set(format_df["team_1"].dropna().astype(str))
    | set(format_df["team_2"].dropna().astype(str))
)

st.header("📋 Match Setup")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox(
        "🏏 Team 1",
        teams
    )

with col2:
    team2_options = [
        team for team in teams
        if team != team1
    ]

    team2 = st.selectbox(
        "🏏 Team 2",
        team2_options
    )


dates = sorted(
    format_df["date"].dropna().dt.date.unique()
)

if dates:
    selected_date = st.date_input(
        "📅 Match Date",
        value=dates[-1],
        min_value=dates[0],
        max_value=dates[-1],
        format="DD/MM/YYYY"
    )
else:
    selected_date = pd.Timestamp.today().date()


venues = sorted(
    format_df["venue"].dropna().astype(str).unique()
)

venue = st.selectbox(
    "📍 Venue",
    venues
)


city_data = format_df[
    format_df["venue"].astype(str) == str(venue)
]

cities = sorted(
    city_data["city"].dropna().astype(str).unique()
)

if cities:
    city = st.selectbox(
        "🏙️ City",
        cities
    )
else:
    city = ""


st.header("🪙 Toss Information")

col3, col4 = st.columns(2)

with col3:
    toss_winner = st.radio(
        "Toss Winner",
        [team1, team2]
    )

with col4:
    toss_decision = st.radio(
        "Toss Decision",
        ["bat", "field"]
    )


if toss_decision == "bat":
    batting_first = toss_winner
else:
    if toss_winner == team1:
        batting_first = team2
    else:
        batting_first = team1

st.info(
    "🏏 Batting First: " + batting_first
)


def get_players(team):

    if players.empty:
        return pd.DataFrame()

    result = players[
        players["team"].astype(str).str.strip()
        == str(team).strip()
    ].copy()

    if result.empty:
        return result

    exact_date = result[
        result["date"].dt.date == selected_date
    ]

    if not exact_date.empty:
        result = exact_date

    venue_result = result[
        result["venue"].astype(str).str.strip()
        == str(venue).strip()
    ]

    if not venue_result.empty:
        result = venue_result

    return result.drop_duplicates(
        subset=["player"]
    )


team1_players = get_players(team1)
team2_players = get_players(team2)


st.markdown("---")

st.header("👥 Players")

player_col1, player_col2 = st.columns(2)

with player_col1:

    st.subheader(team1)

    if team1_players.empty:

        st.warning(
            "No player details found."
        )

    else:

        for number, player in enumerate(
            team1_players["player"].dropna(),
            start=1
        ):

            st.write(
                str(number) + ". " + str(player)
            )


with player_col2:

    st.subheader(team2)

    if team2_players.empty:

        st.warning(
            "No player details found."
        )

    else:

        for number, player in enumerate(
            team2_players["player"].dropna(),
            start=1
        ):

            st.write(
                str(number) + ". " + str(player)
            )


def team_statistics(team):

    history = format_df[
        (
            (format_df["team_1"] == team)
            |
            (format_df["team_2"] == team)
        )
        &
        (
            format_df["date"]
            < pd.Timestamp(selected_date)
        )
    ].sort_values("date")

    if history.empty:

        return {
            "win_rate": 0.5,
            "recent": 0.5,
            "venue": 0.5,
            "city": 0.5
        }

    last = history.iloc[-1]

    if last["team_1"] == team:

        win_rate = last["team_1_win_rate"]
        recent = last["team_1_recent_win_rate"]

    else:

        win_rate = last["team_2_win_rate"]
        recent = last["team_2_recent_win_rate"]


    venue_history = history[
        history["venue"] == venue
    ]

    if venue_history.empty:

        venue_rate = 0.5

    else:

        last_venue = venue_history.iloc[-1]

        if last_venue["team_1"] == team:

            venue_rate = last_venue[
                "team_1_venue_win_rate"
            ]

        else:

            venue_rate = last_venue[
                "team_2_venue_win_rate"
            ]


    city_history = history[
        history["city"] == city
    ]

    if city_history.empty:

        city_rate = 0.5

    else:

        last_city = city_history.iloc[-1]

        if last_city["team_1"] == team:

            city_rate = last_city[
                "team_1_city_win_rate"
            ]

        else:

            city_rate = last_city[
                "team_2_city_win_rate"
            ]


    values = [
        win_rate,
        recent,
        venue_rate,
        city_rate
    ]

    values = [
        0.5 if pd.isna(value) else float(value)
        for value in values
    ]

    return {
        "win_rate": values[0],
        "recent": values[1],
        "venue": values[2],
        "city": values[3]
    }


def h2h_statistics():

    history = format_df[
        (
            (
                (format_df["team_1"] == team1)
                &
                (format_df["team_2"] == team2)
            )
            |
            (
                (format_df["team_1"] == team2)
                &
                (format_df["team_2"] == team1)
            )
        )
        &
        (
            format_df["date"]
            < pd.Timestamp(selected_date)
        )
    ].sort_values("date")

    if history.empty:
        return 0.5, 0.5

    last = history.iloc[-1]

    if last["team_1"] == team1:

        h1 = last["team_1_h2h_win_rate"]
        h2 = last["team_2_h2h_win_rate"]

    else:

        h1 = last["team_2_h2h_win_rate"]
        h2 = last["team_1_h2h_win_rate"]

    if pd.isna(h1):
        h1 = 0.5

    if pd.isna(h2):
        h2 = 0.5

    return float(h1), float(h2)


st.markdown("---")

if st.button(
    "🚀 Predict Match Winner",
    use_container_width=True
):

    try:

        s1 = team_statistics(team1)
        s2 = team_statistics(team2)

        h1, h2 = h2h_statistics()

        team1_toss = 1 if toss_winner == team1 else 0

        team1_batted = 1 if batting_first == team1 else 0

        team1_toss_bat = (
            1
            if (
                toss_winner == team1
                and batting_first == team1
            )
            else 0
        )


        input_data = pd.DataFrame([{

            "match_type": selected_format,

            "team_1": team1,

            "team_2": team2,

            "venue": venue,

            "city": city,

            "toss_winner": toss_winner,

            "toss_decision": toss_decision,

            "team_1_win_rate":
                s1["win_rate"],

            "team_2_win_rate":
                s2["win_rate"],

            "win_rate_difference":
                s1["win_rate"] - s2["win_rate"],

            "team_1_recent_win_rate":
                s1["recent"],

            "team_2_recent_win_rate":
                s2["recent"],

            "recent_form_difference":
                s1["recent"] - s2["recent"],

            "team_1_h2h_win_rate":
                h1,

            "team_2_h2h_win_rate":
                h2,

            "h2h_difference":
                h1 - h2,

            "team_1_won_toss":
                team1_toss,

            "team_1_batted_first":
                team1_batted,

            "team_1_venue_win_rate":
                s1["venue"],

            "team_2_venue_win_rate":
                s2["venue"],

            "venue_win_rate_difference":
                s1["venue"] - s2["venue"],

            "team_1_city_win_rate":
                s1["city"],

            "team_2_city_win_rate":
                s2["city"],

            "city_win_rate_difference":
                s1["city"] - s2["city"],

            "team_1_toss_and_bat":
                team1_toss_bat

        }])


        model_features = list(
            model.feature_names_in_
        )

        input_data = input_data[
            model_features
        ]


        probabilities = model.predict_proba(
            input_data
        )[0]


        team1_probability = 0.0
        team2_probability = 0.0


        for class_value, probability in zip(
            model.classes_,
            probabilities
        ):

            if int(class_value) == 1:

                team1_probability = float(
                    probability
                )

            elif int(class_value) == 0:

                team2_probability = float(
                    probability
                )


        team1_percent = round(
            team1_probability * 100,
            2
        )

        team2_percent = round(
            team2_probability * 100,
            2
        )


        if team1_probability > team2_probability:

            winner = team1

        elif team2_probability > team1_probability:

            winner = team2

        else:

            winner = "Too Close to Call"


        confidence = round(
            max(
                team1_percent,
                team2_percent
            ),
            2
        )


        if confidence >= 70:

            level = "High 🟢"

        elif confidence >= 60:

            level = "Medium 🟡"

        else:

            level = "Low 🔴"


        st.markdown("---")

        st.header("🎯 Prediction Result")

        st.success(
            "🏆 Predicted Winner: "
            + winner
        )


        result1, result2 = st.columns(2)

        with result1:

            st.metric(
                team1 + " Win Probability",
                str(team1_percent) + "%"
            )

            st.progress(
                team1_probability
            )


        with result2:

            st.metric(
                team2 + " Win Probability",
                str(team2_percent) + "%"
            )

            st.progress(
                team2_probability
            )


        st.subheader("📊 Confidence")

        st.metric(
            "Confidence",
            str(confidence) + "%"
        )

        st.write(
            "Confidence Level: **"
            + level
            + "**"
        )


        st.subheader("🏏 Match Information")

        st.write(
            "**Format:** "
            + selected_format
        )

        st.write(
            "**Date:** "
            + selected_date.strftime(
                "%d/%m/%Y"
            )
        )

        st.write(
            "**Team 1:** "
            + team1
        )

        st.write(
            "**Team 2:** "
            + team2
        )

        st.write(
            "**Venue:** "
            + venue
        )

        st.write(
            "**City:** "
            + city
        )

        st.write(
            "**Toss Winner:** "
            + toss_winner
        )

        st.write(
            "**Toss Decision:** "
            + toss_decision
        )

        st.write(
            "**Batting First:** "
            + batting_first
        )


        st.markdown("---")

        st.header("👥 Selected Match Players")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.subheader(team1)

            if team1_players.empty:

                st.info(
                    "No player details available."
                )

            else:

                for number, player in enumerate(
                    team1_players["player"].dropna(),
                    start=1
                ):

                    st.write(
                        str(number)
                        + ". "
                        + str(player)
                    )


        with result_col2:

            st.subheader(team2)

            if team2_players.empty:

                st.info(
                    "No player details available."
                )

            else:

                for number, player in enumerate(
                    team2_players["player"].dropna(),
                    start=1
                ):

                    st.write(
                        str(number)
                        + ". "
                        + str(player)
                    )


    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(error)


st.markdown("---")

st.caption(
    "🏏 Cricket Match Winner Prediction "
    "• IPL • T20I • ODI • Test "
    "• Random Forest"
)