"""
Streamlit data app about judicial nominations.
"""

import altair as alt
import pandas as pd
import streamlit as st

import utils as utl

st.set_page_config(layout="wide", page_title=utl.APP_TITLE)


@st.cache
def load_cached_data():
    return utl.load_data()


@st.cache
def cached_conversion_curves(df, cohort_var):
    return utl.estimate_conversion_curves(df, cohort_var)


## Read data
df = load_cached_data()


## User input widgets on the sidebar
cohort_var = st.sidebar.selectbox("Select cohort Field", utl.COHORT_OPTIONS, index=0)

cohort_level_counts = df[cohort_var].value_counts()
all_cohort_levels = sorted(cohort_level_counts.index)


select_all_levels = st.sidebar.button("Select all cohort levels")
select_top_levels = st.sidebar.button("Select most frequent cohort levels")

select_specific_levels = st.sidebar.multiselect(
    "Select specific cohort levels",
    options=sorted(cohort_level_counts.index),
    default=[],
)

if select_all_levels:
    chosen_levels = all_cohort_levels
elif select_top_levels:
    chosen_levels = list(cohort_level_counts.sort_values(ascending=False).index[:3])
elif select_specific_levels:
    chosen_levels = select_specific_levels
else:
    chosen_levels = []


## Process the impact of user inputs.
df = df[df[cohort_var].isin(chosen_levels)]


## Main page
st.title(utl.APP_TITLE)
st.write(utl.APP_DESCRIPTION)

left_col, right_col = st.beta_columns([1, 1.25])


## Outcome distributions by cohort.
grp = df.groupby([cohort_var, "last_action"])
outcomes = pd.DataFrame(grp.size(), columns=["Count"]).reset_index()

fig = (
    alt.Chart(outcomes)
    .mark_bar()
    .encode(
        x=cohort_var,
        y="Count",
        color="last_action",
        order=alt.Order("last_action", sort="ascending"),
        tooltip=[cohort_var, "last_action", "Count"],
    )
    .properties(
        title=utl.PLOT_TITLES["outcomes"],
    )
)
right_col.altair_chart(fig, use_container_width=True)


## Time series of nominations for each cohort.
grp = df.groupby([cohort_var, "nom_session_month"])
ts_tally = pd.DataFrame(grp.size(), columns=["Count"]).reset_index()

fig = (
    alt.Chart(ts_tally)
    .mark_line()
    .encode(
        x=alt.X(
            "nom_session_month",
            axis=alt.Axis(title=utl.TIME_SERIES_AXIS_TITLE),
            scale=alt.Scale(domain=[1, 25]),
        ),
        y="Count",
        color=alt.Color(cohort_var, legend=None),
        tooltip=[cohort_var, "nom_session_month", "Count"],
    )
    .properties(title=utl.PLOT_TITLES["ts_noms"])
)
left_col.altair_chart(fig, use_container_width=True)


## Time series of confirmations for each cohort.
df2 = df.query("last_action == 'confirmed'")
grp = df2.groupby([cohort_var, "action_session_month"])
ts_tally = pd.DataFrame(grp.size(), columns=["Count"]).reset_index()

fig = (
    alt.Chart(ts_tally)
    .mark_line()
    .encode(
        x=alt.X(
            "action_session_month",
            axis=alt.Axis(title=utl.TIME_SERIES_AXIS_TITLE),
            scale=alt.Scale(domain=[1, 25]),
        ),
        y="Count",
        color=alt.Color(cohort_var, legend=None),
        tooltip=[cohort_var, "action_session_month", "Count"],
    )
    .properties(title=utl.PLOT_TITLES["ts_confirms"])
)
left_col.altair_chart(fig, use_container_width=True)


## Time-to-confirmation model and plot.
# Ignore the actual failures for now.
df_model = cached_conversion_curves(df, cohort_var)

fig = (
    alt.Chart(df_model)
    .mark_line()
    .encode(
        x=alt.X("days_elapsed", axis=alt.Axis(title=utl.CONVERSION_XAXIS_TITLE)),
        y=alt.Y(
            "prob_convert", axis=alt.Axis(format="%", title=utl.CONVERSION_YAXIS_TITLE)
        ),
        color=cohort_var,
        tooltip=[cohort_var, "days_elapsed", "prob_convert"],
    )
    .properties(title=utl.PLOT_TITLES["conversion"])
)

right_col.altair_chart(fig, use_container_width=True)
