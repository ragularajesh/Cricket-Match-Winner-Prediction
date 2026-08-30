import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/cleaned/combined_features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cricket_winner_random_forest.pkl"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("       CRICKET MATCH WINNER - MODEL TRAINING")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)

print("\nDataset shape:", df.shape)

print("\nMatch formats:")
print(df["match_type"].value_counts())

# ============================================================
# FEATURES
# ============================================================

features = [
    "match_type",

    "team_1",
    "team_2",

    "venue",
    "city",

    "toss_winner",
    "toss_decision",

    "team_1_win_rate",
    "team_2_win_rate",

    "win_rate_difference",

    "team_1_recent_win_rate",
    "team_2_recent_win_rate",

    "recent_form_difference",

    "team_1_h2h_win_rate",
    "team_2_h2h_win_rate",

    "h2h_difference",

    "team_1_won_toss",
    "team_1_batted_first",

    "team_1_venue_win_rate",
    "team_2_venue_win_rate",

    "venue_win_rate_difference",

    "team_1_city_win_rate",
    "team_2_city_win_rate",

    "city_win_rate_difference",

    "team_1_toss_and_bat"
]

target = "target"

# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    column
    for column in features
    if column not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")
    print(missing_features)

    raise SystemExit()

# ============================================================
# PREPARE DATA
# ============================================================

X = df[features].copy()
y = df[target].copy()

# Fill categorical missing values
categorical_features = [
    "match_type",
    "team_1",
    "team_2",
    "venue",
    "city",
    "toss_winner",
    "toss_decision"
]

for column in categorical_features:
    X[column] = X[column].fillna("Unknown").astype(str)

# Fill numeric missing values
numeric_features = [
    column
    for column in features
    if column not in categorical_features
]

for column in numeric_features:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

    X[column] = X[column].fillna(
        X[column].median()
    )

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=500,

    max_depth=18,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)

# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)

# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")

# ============================================================
# PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 65)
print("                    MODEL RESULTS")
print("=" * 65)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Team 2",
            "Team 1"
        ]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ============================================================
# FORMAT-WISE TEST ACCURACY
# ============================================================

print("\n" + "=" * 65)
print("              FORMAT-WISE PERFORMANCE")
print("=" * 65)

test_results = X_test.copy()

test_results["actual"] = y_test
test_results["predicted"] = y_pred

for match_type in sorted(
    test_results["match_type"].unique()
):

    subset = test_results[
        test_results["match_type"]
        == match_type
    ]

    format_accuracy = accuracy_score(
        subset["actual"],
        subset["predicted"]
    )

    print(
        f"{match_type:8s}: "
        f"{format_accuracy * 100:.2f}% "
        f"({len(subset)} matches)"
    )

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\n" + "=" * 65)
print("                  MODEL SAVED")
print("=" * 65)

print("\nModel:")
print(MODEL_PATH)

print("\nComplete.")