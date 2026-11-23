
import os
import joblib
import pandas as pd
import streamlit as st


st.title("🏏 Cricket Match Winner Prediction")




        st.subheader("🏏 Match Information")

        st.write("**Format:** " + selected_format)
        st.write("**Date:** " + selected_date.strftime("%d/%m/%Y"))
        st.write("**Team 1:** " + team1)
        st.write("**Team 2:** " + team2)
        st.write("**Venue:** " + venue)
        st.write("**City:** " + city)
        st.write("**Toss Winner:** " + toss_winner)
        st.write("**Toss Decision:** " + toss_decision)
        st.write("**Batting First:** " + batting_first)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("🏏 Cricket Match Winner Prediction • IPL • T20I • ODI • Test • Random Forest")        
st.subheader("🏏 Match Information")

        st.write(
            "**Format:** "
            + selected_format
        )

        st.write(
            "**Date:** "
            + selected_date.strftime("%d/%m/%Y")
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


        if selected_format == "IPL":
                    # Display selected players.

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


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🏏 Cricket Match Winner Prediction "
    "• IPL • T20I • ODI • Test "
    "• Random Forest"
)
