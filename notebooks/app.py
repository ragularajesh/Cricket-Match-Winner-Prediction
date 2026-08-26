import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Cricket Winner Predictor", page_icon="🏏", layout="centered")

st.title("🏏 Cricket Match Winner Predictor")
st.markdown("Predict match win probabilities using the **14-Feature Logistic Regression Model**.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_cricket_model_14f.pkl")
    features = joblib.load("model_features.pkl")
    df_enriched = pd.read_csv("df_enriched.csv")
    return model, features, df_enriched

try:
    model, features, df_enriched = load_artifacts()
except Exception as e:
    st.error("Missing model or data files! Make sure 'best_cricket_model_14f.pkl', 'model_features.pkl', and 'df_enriched.csv' are in this directory.")
    st.stop()

all_teams = sorted(list(set(df_enriched['team_1'].unique()).union(set(df_enriched['team_2'].unique()))))
all_venues = sorted(df_enriched['venue'].dropna().unique().tolist())

st.header("📋 Match Setup")
col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Select Team 1", options=all_teams, index=0)
remaining_teams = [t for t in all_teams if t != team1]
with col2:
    team2 = st.selectbox("Select Team 2", options=remaining_teams, index=0)

venue = st.selectbox("Select Venue", options=all_venues)

col3, col4 = st.columns(2)
with col3:
    toss_winner = st.radio("Toss Winner", options=[team1, team2])
with col4:
    batted_first = st.radio("Team Batting First", options=[team1, team2])

def predict_match_ui(t1, t2, toss_w, bat_1st, ven):
    def get_latest_stats(team, venue_name):
        team_matches = df_enriched[(df_enriched['team_1'] == team) | (df_enriched['team_2'] == team)]
        if team_matches.empty:
            return {'win_rate': 0.5, 'recent_win_rate': 0.5, 'venue_win_rate': 0.5}
        last_match = team_matches.iloc[-1]
        wr = last_match['team_1_win_rate'] if last_match['team_1'] == team else last_match['team_2_win_rate']
        rwr = last_match['team_1_recent_win_rate'] if last_match['team_1'] == team else last_match['team_2_recent_win_rate']

        venue_matches = team_matches[team_matches['venue'] == venue_name]
        if not venue_matches.empty:
            v_last = venue_matches.iloc[-1]
            vwr = v_last['team_1_venue_win_rate'] if v_last['team_1'] == team else v_last['team_2_venue_win_rate']
        else:
            vwr = 0.5
        return {'win_rate': wr, 'recent_win_rate': rwr, 'venue_win_rate': vwr}

    t1_s = get_latest_stats(t1, ven)
    t2_s = get_latest_stats(t2, ven)

    h2h = df_enriched[((df_enriched['team_1'] == t1) & (df_enriched['team_2'] == t2)) | ((df_enriched['team_1'] == t2) & (df_enriched['team_2'] == t1))]
    if not h2h.empty:
        h2h_last = h2h.iloc[-1]
        h2h_1 = h2h_last['team_1_h2h_win_rate'] if h2h_last['team_1'] == t1 else h2h_last['team_2_h2h_win_rate']
        h2h_2 = h2h_last['team_2_h2h_win_rate'] if h2h_last['team_1'] == t1 else h2h_last['team_1_h2h_win_rate']
    else:
        h2h_1, h2h_2 = 0.5, 0.5

    match_data = {
        'team_1_win_rate': t1_s['win_rate'],
        'team_2_win_rate': t2_s['win_rate'],
        'win_rate_difference': t1_s['win_rate'] - t2_s['win_rate'],
        'team_1_recent_win_rate': t1_s['recent_win_rate'],
        'team_2_recent_win_rate': t2_s['recent_win_rate'],
        'recent_form_difference': t1_s['recent_win_rate'] - t2_s['recent_win_rate'],
        'team_1_h2h_win_rate': h2h_1,
        'team_2_h2h_win_rate': h2h_2,
        'h2h_difference': h2h_1 - h2h_2,
        'team_1_won_toss': 1 if toss_w == t1 else 0,
        'team_1_batted_first': 1 if bat_1st == t1 else 0,
        'team_1_venue_win_rate': t1_s['venue_win_rate'],
        'team_2_venue_win_rate': t2_s['venue_win_rate'],
        'venue_win_rate_difference': t1_s['venue_win_rate'] - t2_s['venue_win_rate'],
    }

    input_df = pd.DataFrame([match_data])[features]
    return model.predict_proba(input_df)[0]

st.markdown("---")
if st.button("🚀 Predict Match Outcome", use_container_width=True):
    probs = predict_match_ui(team1, team2, toss_winner, batted_first, venue)
    t1_prob, t2_prob = round(probs[0] * 100, 2), round(probs[1] * 100, 2)
    winner = team1 if t1_prob > t2_prob else team2

    st.subheader("🎯 Result")
    st.success(f"🏆 Predicted Winner: **{winner}**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(f"{team1} Win Probability", f"{t1_prob}%")
        st.progress(t1_prob / 100)
    with col_b:
        st.metric(f"{team2} Win Probability", f"{t2_prob}%")
        st.progress(t2_prob / 100)
