import streamlit as st
import pandas as pd

import plotly.express as px


def plot_classification_overall_winrate(df: pd.DataFrame):
    if df.empty:
        st.warning("No evaluation data to plot.")
        return

    df = df.copy()
    df["evaluation_date"] = pd.to_datetime(df["evaluation_date"]).dt.normalize()

    df_plot = (
        df.groupby(["model", "evaluation_date"])["correct"]
        .agg(daily_wins="sum", daily_total="count")
        .reset_index()
        .sort_values(["model", "evaluation_date"])
    )

    df_plot["cumulative_wins"] = df_plot.groupby("model")["daily_wins"].cumsum()
    df_plot["cumulative_total"] = df_plot.groupby("model")["daily_total"].cumsum()
    df_plot["winrate"] = (
        df_plot["cumulative_wins"] / df_plot["cumulative_total"] * 100
    ).round(2)

    fig = px.line(
        df_plot,
        x="evaluation_date",
        y="winrate",
        color="model",
        markers=True,
        line_shape="linear",
        hover_data={
            "evaluation_date": "|%Y-%m-%d",
            "winrate": ":.2f}%",
            "cumulative_wins": True,
            "cumulative_total": True,
        },
        template="plotly_dark",
        labels={"winrate": "Cumulative Win Rate (%)", "evaluation_date": "Date"},
    )

    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="rgba(255,255,255,0.35)",
        annotation_text="50% baseline",
    )

    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    fig.update_yaxes(ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True)

    summary = (
        df.groupby("model")["correct"].agg(total="count", wins="sum").reset_index()
    )
    summary["winrate"] = (summary["wins"] / summary["total"] * 100).round(2)
    summary = summary.sort_values("winrate", ascending=False)

    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        st.metric(
            label=row["model"],
            value=f"{row['winrate']}%",
            delta=f"{row['wins']}/{row['total']} correct",
            delta_color="normal" if row["winrate"] >= 50 else "inverse",
        )


def plot_regression_overall_error(df: pd.DataFrame):
    if df.empty:
        st.warning("No evaluation data to plot.")
        return

    df = df.copy()
    df["evaluation_date"] = pd.to_datetime(df["evaluation_date"]).dt.normalize()

    df_plot = (
        df.groupby(["model", "evaluation_date"])["abs_error"]
        .agg(daily_error="mean")
        .reset_index()
        .sort_values(["model", "evaluation_date"])
    )

    df_plot["cumulative_error"] = df_plot.groupby("model")["daily_error"].cumsum() / (
        df_plot.groupby("model").cumcount() + 1
    )

    fig = px.line(
        df_plot,
        x="evaluation_date",
        y="cumulative_error",
        color="model",
        markers=True,
        line_shape="linear",
        hover_data={"evaluation_date": "|%Y-%m-%d", "cumulative_error": ":.4f"},
        template="plotly_dark",
        labels={"cumulative_error": "Cumulative MAE", "evaluation_date": "Date"},
    )

    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    summary = df.groupby("model")["abs_error"].agg(mean_error="mean").reset_index()
    summary = summary.sort_values("mean_error")

    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        st.metric(
            label=row["model"],
            value=f"{row['mean_error']:.4f}",
            delta=None,
            delta_color="normal",
        )
