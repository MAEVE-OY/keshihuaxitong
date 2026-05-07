from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="情感分析 | 美食与节庆文化对比",
    page_icon="💬",
    layout="wide",
)
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "all_data.csv"

CATEGORY_LABELS = {
    "food": "美食",
    "festival": "节庆文化",
    "美食": "美食",
    "节庆": "节庆文化",
    "节庆文化": "节庆文化",
}

CATEGORY_ORDER = ["美食", "节庆文化"]
SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_LABELS_CN = {
    "positive": "正向",
    "neutral": "中性",
    "negative": "负向",
}
SENTIMENT_COLORS = {
    "positive": "#fdd741",
    "neutral": "#8fedc2",
    "negative": "#2fb4af",
}
CATEGORY_COLORS = {
    "美食": "#F4A261",
    "节庆文化": "#5B8DEF",
}

COUNTRY_CANDIDATES = [
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "Singapore",
    "India",
]


@st.cache_data(show_spinner="正在读取情感分析数据...")
def load_data(uploaded_file=None, default_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    source = uploaded_file if uploaded_file is not None else default_path
    df = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if "tweet_created_at" in df.columns:
        df["tweet_created_at"] = pd.to_datetime(df["tweet_created_at"], errors="coerce", utc=True)
        df["tweet_date"] = df["tweet_created_at"].dt.date

    numeric_cols = [
        "year",
        "month",
        "compound",
        "neg",
        "neu",
        "pos",
        "rt_count",
        "like_count",
        "reply_count",
        "quote_count",
        "engagement",
        "followers_count",
        "following_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "year" not in df.columns and "tweet_created_at" in df.columns:
        df["year"] = df["tweet_created_at"].dt.year
    if "month" not in df.columns and "tweet_created_at" in df.columns:
        df["month"] = df["tweet_created_at"].dt.month

    if "engagement" not in df.columns:
        interaction_cols = [c for c in ["rt_count", "like_count", "reply_count", "quote_count"] if c in df.columns]
        df["engagement"] = df[interaction_cols].sum(axis=1) if interaction_cols else 0
    df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(0)

    for col in ["category", "sentiment_label", "country", "city", "text", "clean_text", "final_topic"]:
        if col in df.columns:
            df[col] = df[col].fillna("未知").astype(str).str.strip()

    if "category" in df.columns:
        df["category_key"] = df["category"].str.lower().str.strip()
        df["category_label"] = df["category_key"].map(CATEGORY_LABELS).fillna(df["category"].map(CATEGORY_LABELS)).fillna(df["category"])
    else:
        df["category_label"] = "未知"

    if "sentiment_label" in df.columns:
        df["sentiment_label"] = df["sentiment_label"].str.lower().str.strip()
        df = df[df["sentiment_label"].isin(SENTIMENT_ORDER)].copy()

    if "compound" in df.columns:
        df["compound"] = pd.to_numeric(df["compound"], errors="coerce")

    return df


def ensure_columns(df: pd.DataFrame, required_cols: list[str]) -> bool:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.warning("缺少字段：" + "、".join(missing))
        return False
    return True


def format_percent(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


def format_score(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.3f}"


def style_chart(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, SimHei, Arial", size=13, color="#334155"),
        title=dict(font=dict(size=18, color="#1E293B"), x=0.02, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
            bgcolor="rgba(255,255,255,0)",
        ),
        margin=dict(l=30, r=30, t=78, b=45),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)", zeroline=False)
    return fig


def render_metric_pair(title: str, food_value: str, festival_value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-row">
                <div class="metric-half food-card">
                    <div class="metric-label">Food 美食</div>
                    <div class="metric-value">{food_value}</div>
                </div>
                <div class="metric-half festival-card">
                    <div class="metric-label">Festival 节庆</div>
                    <div class="metric-value">{festival_value}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calc_sentiment_share(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["category_label"].isin(CATEGORY_ORDER)].copy()
    count_df = (
        base.groupby(["category_label", "sentiment_label"], as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
    )
    full_index = pd.MultiIndex.from_product([CATEGORY_ORDER, SENTIMENT_ORDER], names=["category_label", "sentiment_label"])
    count_df = count_df.set_index(["category_label", "sentiment_label"]).reindex(full_index, fill_value=0).reset_index()
    total_df = count_df.groupby("category_label")["tweet_count"].transform("sum")
    count_df["share"] = np.where(total_df > 0, count_df["tweet_count"] / total_df, 0)
    count_df["sentiment_cn"] = count_df["sentiment_label"].map(SENTIMENT_LABELS_CN)
    return count_df


def get_share(sentiment_share: pd.DataFrame, category: str, sentiment: str) -> float:
    rows = sentiment_share[
        (sentiment_share["category_label"] == category) &
        (sentiment_share["sentiment_label"] == sentiment)
    ]
    return float(rows["share"].iloc[0]) if not rows.empty else np.nan


def display_text_table(data: pd.DataFrame, max_rows: int = 8):
    display_cols = [c for c in ["text", "country", "city", "compound", "engagement"] if c in data.columns]
    if not display_cols:
        st.info("当前数据缺少可展示的推文字段。")
        return
    table = data[display_cols].head(max_rows).copy()
    rename_map = {
        "text": "推文文本",
        "country": "国家",
        "city": "城市",
        "compound": "情感得分",
        "engagement": "互动量",
    }
    table = table.rename(columns=rename_map)
    if "情感得分" in table.columns:
        table["情感得分"] = table["情感得分"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    st.dataframe(table, use_container_width=True, hide_index=True, height=360)


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #FFF7ED 0%, #EFF6FF 52%, #F8FAFC 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        line-height: 1.25;
        font-weight: 800;
        color: #172554;
        margin-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #1E293B;
        margin-top: 1.5rem;
        margin-bottom: 0.35rem;
    }
    .metric-card {
        min-height: 125px;
        padding: 1rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
        margin-bottom: 0.75rem;
    }
    .metric-title {
        font-size: 0.98rem;
        font-weight: 750;
        color: #0F172A;
        margin-bottom: 0.8rem;
    }
    .metric-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
    }
    .metric-half {
        border-radius: 16px;
        padding: 0.72rem 0.75rem;
    }
    .food-card {
        background: linear-gradient(135deg, rgba(244,162,97,0.18), rgba(255,247,237,0.9));
        border: 1px solid rgba(244,162,97,0.28);
    }
    .festival-card {
        background: linear-gradient(135deg, rgba(91,141,239,0.17), rgba(239,246,255,0.92));
        border: 1px solid rgba(91,141,239,0.28);
    }
    .metric-label {
        color: #64748B;
        font-size: 0.76rem;
        font-weight: 650;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.52rem;
        font-weight: 850;
        margin-top: 0.12rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">情感分析：美食文化与节庆文化的情绪传播差异</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("数据与筛选")
    df = load_data()

    if not ensure_columns(df, ["category_label", "sentiment_label", "compound"]):
        st.stop()

    years_available = sorted(df["year"].dropna().astype(int).unique().tolist()) if "year" in df.columns else []
    if years_available:
        selected_years = st.slider(
            "年份范围",
            min_value=min(years_available),
            max_value=max(years_available),
            value=(min(years_available), max(years_available)),
        )
    else:
        selected_years = None

    country_options = []
    if "country" in df.columns:
        existing_countries = sorted([c for c in df["country"].dropna().unique().tolist() if c and c != "未知"])
        preferred = [c for c in COUNTRY_CANDIDATES if c in existing_countries]
        other = [c for c in existing_countries if c not in preferred]
        country_options = ["全部"] + preferred + other
        selected_country = st.selectbox("国家", country_options, index=0)
    else:
        selected_country = "全部"

    sentiment_options = ["全部"] + [SENTIMENT_LABELS_CN[s] for s in SENTIMENT_ORDER]
    selected_sentiment_cn = st.selectbox("情感类型", sentiment_options, index=0)
    selected_sentiment = None
    if selected_sentiment_cn != "全部":
        selected_sentiment = {v: k for k, v in SENTIMENT_LABELS_CN.items()}[selected_sentiment_cn]


filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["category_label"].isin(CATEGORY_ORDER)].copy()

if selected_years and "year" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["year"] >= selected_years[0]) &
        (filtered_df["year"] <= selected_years[1])
    ]
if selected_country != "全部" and "country" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["country"] == selected_country]
if selected_sentiment and "sentiment_label" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["sentiment_label"] == selected_sentiment]

if filtered_df.empty:
    st.info("当前筛选条件下没有可展示的数据，请调整年份、国家或情感类型筛选。")
    st.stop()

st.markdown('<div class="section-title">1. 情感对比概况</div>', unsafe_allow_html=True)

sentiment_share = calc_sentiment_share(filtered_df)
food_positive = get_share(sentiment_share, "美食", "positive")
festival_positive = get_share(sentiment_share, "节庆文化", "positive")
food_neutral = get_share(sentiment_share, "美食", "neutral")
festival_neutral = get_share(sentiment_share, "节庆文化", "neutral")
food_negative = get_share(sentiment_share, "美食", "negative")
festival_negative = get_share(sentiment_share, "节庆文化", "negative")

avg_score = (
    filtered_df.groupby("category_label")["compound"]
    .mean()
    .reindex(CATEGORY_ORDER)
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_pair("正向情绪占比", format_percent(food_positive), format_percent(festival_positive))
with m2:
    render_metric_pair("中性情绪占比", format_percent(food_neutral), format_percent(festival_neutral))
with m3:
    render_metric_pair("负向情绪占比", format_percent(food_negative), format_percent(festival_negative))
with m4:
    render_metric_pair("平均情感得分", format_score(avg_score.get("美食")), format_score(avg_score.get("节庆文化")))

st.divider()
st.markdown('<div class="section-title">2. 美食与节庆的情感结构对比</div>', unsafe_allow_html=True)

if ensure_columns(filtered_df, ["category_label", "sentiment_label"]):
    fig_structure = px.bar(
        sentiment_share,
        x="category_label",
        y="share",
        color="sentiment_label",
        text=sentiment_share["share"].map(lambda x: f"{x:.1%}" if x > 0 else ""),
        category_orders={"category_label": CATEGORY_ORDER, "sentiment_label": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        title="美食类与节庆类推文的情感结构对比",
        labels={"category_label": "内容类别", "share": "情感比例", "sentiment_label": "情感类型"},
    )
    fig_structure.update_traces(
        textposition="inside",
        hovertemplate="%{x}<br>情感类型：%{legendgroup}<br>比例：%{y:.1%}<extra></extra>",
    )
    fig_structure.update_layout(
        barmode="stack",
        yaxis_tickformat=".0%",
        legend_title_text="情感类型",
    )
    st.plotly_chart(style_chart(fig_structure, 450), use_container_width=True)

st.divider()
st.markdown('<div class="section-title">3. 美食与节庆的情感得分分布</div>', unsafe_allow_html=True)

if ensure_columns(filtered_df, ["category_label", "compound"]):
    box_df = filtered_df.dropna(subset=["compound"])
    fig_box = px.box(
        box_df,
        x="category_label",
        y="compound",
        color="category_label",
        points="outliers",
        category_orders={"category_label": CATEGORY_ORDER},
        color_discrete_map=CATEGORY_COLORS,
        title="美食类与节庆类推文的 compound 情感得分分布",
        labels={"category_label": "内容类别", "compound": "compound 情感得分"},
    )
    fig_box.add_hline(y=0, line_dash="dash", line_color="#64748B")
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(style_chart(fig_box, 450), use_container_width=True)

st.divider()
st.markdown('<div class="section-title">4. 时间维度下的情感变化</div>', unsafe_allow_html=True)

if ensure_columns(filtered_df, ["year", "category_label", "compound"]):
    year_score = (
        filtered_df.dropna(subset=["year", "compound"])
        .groupby(["year", "category_label"], as_index=False)["compound"]
        .mean()
        .sort_values(["year", "category_label"])
    )
    year_score["year"] = year_score["year"].astype(int)
    fig_year = px.line(
        year_score,
        x="year",
        y="compound",
        color="category_label",
        markers=True,
        category_orders={"category_label": CATEGORY_ORDER},
        color_discrete_map=CATEGORY_COLORS,
        title="美食类与节庆类推文年度平均情感得分变化",
        labels={"year": "年份", "compound": "平均 compound", "category_label": "内容类别"},
    )
    fig_year.add_hline(y=0, line_dash="dash", line_color="#64748B")
    fig_year.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(style_chart(fig_year, 460), use_container_width=True)

st.divider()
st.markdown('<div class="section-title">5. 空间维度下的情感差异</div>', unsafe_allow_html=True)

if ensure_columns(filtered_df, ["country", "category_label", "compound"]):
    country_base = filtered_df.dropna(subset=["compound"]).copy()
    country_count = (
        country_base.groupby(["country", "category_label"], as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
    )
    country_score = (
        country_base.groupby(["country", "category_label"], as_index=False)["compound"]
        .mean()
        .rename(columns={"compound": "avg_compound"})
    )
    country_score = country_score.merge(country_count, on=["country", "category_label"], how="left")

    country_totals = (
        country_base.groupby("country", as_index=False)
        .size()
        .rename(columns={"size": "total_count"})
        .sort_values("total_count", ascending=False)
    )
    preferred_countries = [c for c in COUNTRY_CANDIDATES if c in country_totals["country"].tolist()]
    if len(preferred_countries) < 6:
        extra_countries = [c for c in country_totals["country"].tolist() if c not in preferred_countries and c != "未知"]
        preferred_countries += extra_countries[: 6 - len(preferred_countries)]

    heat_df = country_score[country_score["country"].isin(preferred_countries)].copy()
    heat_pivot = heat_df.pivot(index="country", columns="category_label", values="avg_compound")
    heat_pivot = heat_pivot.reindex(index=preferred_countries, columns=CATEGORY_ORDER)

    if heat_pivot.dropna(how="all").empty:
        st.info("当前筛选条件下没有足够的国家情感得分数据。")
    else:
        fig_country_heat = px.imshow(
            heat_pivot,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="重点国家中美食类与节庆类推文的平均情感得分对比",
            labels=dict(x="内容类别", y="国家", color="平均 compound"),
        )
        fig_country_heat.update_xaxes(side="top")
        st.plotly_chart(style_chart(fig_country_heat, 440), use_container_width=True)

st.divider()
st.markdown('<div class="section-title">6. 代表性推文文本对比</div>', unsafe_allow_html=True)

if ensure_columns(filtered_df, ["category_label", "sentiment_label", "compound"]):
    rep_sentiment_cn = st.radio(
        "选择代表推文情感类型",
        [SENTIMENT_LABELS_CN[s] for s in SENTIMENT_ORDER],
        horizontal=True,
        index=0,
    )
    rep_sentiment = {v: k for k, v in SENTIMENT_LABELS_CN.items()}[rep_sentiment_cn]

    top_n_tweets = st.slider(
        "选择每类显示的代表推文数量",
        min_value=3,
        max_value=30,
        value=10,
        step=1,
    )

    rep_df = filtered_df[filtered_df["sentiment_label"] == rep_sentiment].copy()

    if rep_sentiment == "positive":
        rep_df = rep_df.sort_values(["compound", "engagement"], ascending=[False, False])
    elif rep_sentiment == "negative":
        rep_df = rep_df.sort_values(["compound", "engagement"], ascending=[True, False])
    else:
        rep_df["abs_compound"] = rep_df["compound"].abs()
        rep_df = rep_df.sort_values(["abs_compound", "engagement"], ascending=[True, False])

    left, right = st.columns(2)

    with left:
        st.markdown("#### Food 美食代表推文")
        display_text_table(
            rep_df[rep_df["category_label"] == "美食"],
            max_rows=top_n_tweets
        )

    with right:
        st.markdown("#### Festival 节庆代表推文")
        display_text_table(
            rep_df[rep_df["category_label"] == "节庆文化"],
            max_rows=top_n_tweets
        )