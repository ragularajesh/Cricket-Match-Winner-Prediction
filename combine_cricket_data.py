import os
import json
import pandas as pd

# ============================================================
# CRICKET DATA COMBINER
# T20I + ODI + TEST + IPL
# ============================================================

BASE_DIR = "data/raw"

FOLDERS = {
    "T20I": "t20i_json",
    "ODI": "odi_json",
    "Test": "test_json",
    "IPL": "ipl_json",
}

OUTPUT_DIR = "data/cleaned"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "combined_matches.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_matches = []

print("=" * 60)
print("       CRICKET MATCH DATA COMBINER")
print("=" * 60)


# ============================================================
# READ JSON FILES
# ============================================================

for match_format, folder in FOLDERS.items():

    folder_path = os.path.join(
        BASE_DIR,
        folder
    )

    print()
    print("-" * 60)
    print("Reading:", match_format)
    print("Folder:", folder_path)

    if not os.path.exists(folder_path):

        print("WARNING: Folder not found")
        continue

    json_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(".json")
    ]

    print("JSON files:", len(json_files))

    count = 0

    for filename in json_files:

        filepath = os.path.join(
            folder_path,
            filename
        )

        try:

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            info = data.get("info", {})

            teams = info.get(
                "teams",
                []
            )

            if len(teams) < 2:
                continue

            team1 = teams[0]
            team2 = teams[1]

            dates = info.get(
                "dates",
                []
            )

            if dates:

                match_date = dates[0]

            else:

                match_date = None

            venue = info.get(
                "venue",
                None
            )

            city = info.get(
                "city",
                None
            )

            toss = info.get(
                "toss",
                {}
            )

            toss_winner = toss.get(
                "winner",
                None
            )

            toss_decision = toss.get(
                "decision",
                None
            )

            outcome = info.get(
                "outcome",
                {}
            )

            winner = outcome.get(
                "winner",
                None
            )

            result = outcome.get(
                "result",
                None
            )

            result_winner = outcome.get(
                "winner",
                None
            )

            # ------------------------------------------------
            # MATCH TYPE FROM JSON
            # ------------------------------------------------

            json_match_type = info.get(
                "match_type",
                None
            )

            if json_match_type:

                json_match_type = str(
                    json_match_type
                ).upper()

            # ------------------------------------------------
            # STORE MATCH
            # ------------------------------------------------

            match = {

                "match_id":
                    os.path.splitext(filename)[0],

                "match_type":
                    match_format,

                "json_match_type":
                    json_match_type,

                "date":
                    match_date,

                "team_1":
                    team1,

                "team_2":
                    team2,

                "winner":
                    winner,

                "result":
                    result,

                "toss_winner":
                    toss_winner,

                "toss_decision":
                    toss_decision,

                "venue":
                    venue,

                "city":
                    city
            }

            all_matches.append(match)

            count += 1

        except Exception as error:

            print(
                "ERROR:",
                filename,
                "->",
                error
            )

    print(
        "Successfully processed:",
        count
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

print()
print("=" * 60)
print("Creating combined dataframe...")
print("=" * 60)

df = pd.DataFrame(
    all_matches
)


# ============================================================
# CLEAN DATA
# ============================================================

if not df.empty:

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["team_1"] = (
        df["team_1"]
        .astype(str)
        .str.strip()
    )

    df["team_2"] = (
        df["team_2"]
        .astype(str)
        .str.strip()
    )

    df["winner"] = (
        df["winner"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["match_type"] = (
        df["match_type"]
        .astype(str)
        .str.strip()
    )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("              DATASET CREATED")
print("=" * 60)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("Total matches:")
print(len(df))

print()
print("Dataset shape:")
print(df.shape)

print()
print("Matches by format:")
print(
    df["match_type"]
    .value_counts()
)

print()
print("Columns:")
print(
    df.columns.tolist()
)

print()
print("First 5 matches:")
print(
    df.head()
)

print()
print("=" * 60)
print("              COMPLETE")
print("=" * 60)