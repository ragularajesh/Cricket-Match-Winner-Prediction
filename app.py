import os
import joblib
import pandas as pd
import streamlit as st
import requests
import plotly.express as px
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from streamlit.errors import StreamlitSecretNotFoundError

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


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )
    return data


@st.cache_data
def load_players():
    if os.path.exists(PLAYERS_PATH):
        data = pd.read_csv(PLAYERS_PATH)
        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )
        return data

    return pd.DataFrame()


model = load_model()
df = load_data()
players = load_players()


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🏏 Cricket Match Winner Prediction")
st.write("IPL • T20I • ODI • Test")

# Light theme with black text
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; color: #000000; }
    p, label, span, div, li { color: #000000; }
    h1, h2, h3, h4, h5, h6 { color: #000000 !important; }
    div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #d6dce5; padding: 14px; border-radius: 14px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #d6dce5 !important; }
</style>
""", unsafe_allow_html=True)
