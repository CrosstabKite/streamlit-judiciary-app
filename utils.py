"""
Common code and config for all the data app prototypes.
"""

import pandas as pd
import convoys.utils as cutl
from convoys.multi import KaplanMeier

try:
    import plotly.express as px
except:
    pass


POTUS_TERMS = {
    "Clinton": (pd.Timestamp(1993, 1, 20), pd.Timestamp(2001, 1, 20)),
    "Bush": (pd.Timestamp(2001, 1, 20), pd.Timestamp(2009, 1, 20)),
    "Obama": (pd.Timestamp(2009, 1, 20), pd.Timestamp(2017, 1, 20)),
    "Trump": (pd.Timestamp(2017, 1, 20), pd.Timestamp(2100, 1, 1)),
}

DATE_DATA_DOWNLOADED = pd.Timestamp(2020, 10, 21)

APP_TITLE = "Judicial Nominations"

APP_DESCRIPTION = (
    "The nomination of Amy Coney Barrett to the US Supreme Court has brought the topic "
    "of presidential judicial nominations and Senate confirmations back to the fore. "
    "This app uses basic conversion analysis to explore the success rates and "
    "time-to-confirmation of presidential judicial nominations, broken down by a "
    "cohort variable of your choice."
)

COHORT_OPTIONS = ("congress", "president", "residence", "role")

MAX_SESSION_DAYS = 365 + 366  # Max number of days in a Congressional session.

PLOT_TITLES = {
    "outcomes": "Outcome distribution",
    "ts_noms": "Nominations over time",
    "ts_confirms": "Confirmations over time",
    "conversion": "Time to Confirmation",
}

TIME_SERIES_AXIS_TITLE = "Month of Congressional Session"
CONVERSION_XAXIS_TITLE = "Days Since Nomination"
CONVERSION_YAXIS_TITLE = "Probability of Confirmation"


def lookup_potus(date):
    for potus, (start_date, end_date) in POTUS_TERMS.items():
        if date >= start_date and date < end_date:
            return potus
    return None


def session_month_index(start_date, end_date):
    session_month = 12 * (end_date.dt.year - start_date.dt.year) + end_date.dt.month
    return session_month


def decide_confirmation_date(row):
    if row["last_action"] in ("confirmed"):
        return row["date_last_action"]
    else:
        return None


def load_data():
    """"""
    df = pd.read_csv(
        "judicial_nominations.csv",
        parse_dates=["date_last_action", "date_received"],
        usecols=[
            "url",
            "congress",
            "residence",
            "role",
            "last_action",
            "date_last_action",
            "date_received",
        ],
    )

    df = df.dropna(axis="rows", subset=["last_action"], how="any")

    df["president"] = df["date_received"].apply(lambda x: lookup_potus(x))

    congress_year_start = 2001 + 2 * (df["congress"] - 107)
    congress_year_end = congress_year_start + 2

    df["date_congress_start"] = congress_year_start.apply(
        lambda x: pd.Timestamp(x, 1, 3)
    )

    df["date_congress_end"] = congress_year_end.apply(lambda x: pd.Timestamp(x, 1, 3))

    df["nom_session_month"] = session_month_index(
        df["date_congress_start"], df["date_received"]
    )

    df["action_session_month"] = session_month_index(
        df["date_congress_start"], df["date_last_action"]
    )

    df["congress"] = df["congress"].astype("str")

    df["date_confirmed"] = df.apply(decide_confirmation_date, axis=1)
    df["date_censored"] = df["date_congress_end"].apply(
        lambda x: min(x, DATE_DATA_DOWNLOADED)
    )

    return df


def estimate_conversion_curves(df, cohort_var):
    """"""
    df2 = df.query("last_action != 'withdrawn'")

    if len(df2) > 0:
        _, group_labels, (ix_group, did_convert, time_to_convert) = cutl.get_arrays(
            df2,
            groups=cohort_var,
            created="date_received",
            converted="date_confirmed",
            now="date_censored",
            unit="days",
        )

        model = KaplanMeier()
        model.fit(ix_group, did_convert, time_to_convert)

        time_grid = list(range(0, MAX_SESSION_DAYS + 1))
        df_model = pd.DataFrame(
            {g: model.predict(i, time_grid) for i, g in enumerate(group_labels)}
        )
        df_model.index.name = "days_elapsed"
        df_model = df_model.reset_index()
        df_model = df_model.melt(
            id_vars=["days_elapsed"], var_name=cohort_var, value_name="prob_convert"
        )

    else:
        df_model = pd.DataFrame(columns=[cohort_var, "days_elapsed", "prob_convert"])

    return df_model
