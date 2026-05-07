import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="总览页 | 社媒传播数据分析",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "all_data.csv"

CATEGORY_LABELS = {
    "food": "美食",
    "festival": "节庆文化",
}

INTERACTION_LABELS = {
    "like_count": "点赞",
    "rt_count": "转发",
    "reply_count": "评论/回复",
    "quote_count": "引用",
}


@st.cache_data(show_spinner="正在读取数据...")
def load_data(uploaded_file=None, default_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    source = uploaded_file if uploaded_file is not None else default_path
    df = pd.read_csv(source, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if "tweet_created_at" in df.columns:
        df["tweet_created_at"] = pd.to_datetime(df["tweet_created_at"], errors="coerce", utc=True)
        df["tweet_date"] = df["tweet_created_at"].dt.date

    numeric_cols = [
        "year",
        "month",
        "followers_count",
        "following_count",
        "rt_count",
        "like_count",
        "reply_count",
        "quote_count",
        "engagement",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "engagement" not in df.columns:
        interaction_cols = [c for c in ["rt_count", "like_count", "reply_count", "quote_count"] if c in df.columns]
        df["engagement"] = df[interaction_cols].sum(axis=1) if interaction_cols else 0

    if "year" not in df.columns and "tweet_created_at" in df.columns:
        df["year"] = df["tweet_created_at"].dt.year

    if "month" not in df.columns and "tweet_created_at" in df.columns:
        df["month"] = df["tweet_created_at"].dt.month

    if "year_month" not in df.columns and "tweet_created_at" in df.columns:
        df["year_month"] = df["tweet_created_at"].dt.to_period("M").astype(str)

    for col in ["country", "city", "category", "sentiment_label", "final_topic", "text", "clean_text"]:
        if col in df.columns:
            df[col] = df[col].fillna("未知").astype(str).str.strip()

    if "category" in df.columns:
        df["category_label"] = df["category"].map(CATEGORY_LABELS).fillna(df["category"])

    return df


def format_number(value) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 100_000_000:
        return f"{value / 100_000_000:.2f}亿"
    if abs(value) >= 10_000:
        return f"{value / 10_000:.2f}万"
    return f"{value:,.0f}"


def normalize_country_name(country: str) -> str:
    if pd.isna(country):
        return "Unknown"
    value = str(country).strip()
    value = re.sub(r"\s*\([^)]*\)\s*", "", value).strip()
    mapping = {
        "The United States": "United States",
        "United States of America": "United States",
        "USA": "United States",
        "US": "United States",
        "UK": "United Kingdom",
        "The United Kingdom": "United Kingdom",
        "UAE": "United Arab Emirates",
        "Russia": "Russian Federation",
        "South Korea": "Korea, Republic of",
    }
    return mapping.get(value, value)


def ensure_columns(df: pd.DataFrame, required_cols: list[str]) -> bool:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.warning("缺少字段：" + "、".join(missing))
        return False
    return True


def add_empty_state(message: str):
    st.info(message)


def category_order(df: pd.DataFrame) -> list[str]:
    preferred = ["美食", "节庆文化"]
    existing = df["category_label"].dropna().unique().tolist() if "category_label" in df.columns else []
    return [x for x in preferred if x in existing] + [x for x in existing if x not in preferred]


st.title("社媒传播数据总览")
st.caption("围绕数据规模、类别构成、时间趋势、空间分布与传播互动进行一屏式概览。")

with st.sidebar:
    st.header("数据与筛选")
    df = load_data()

    if "category_label" in df.columns:
        category_options = category_order(df)
        selected_categories = st.multiselect("内容类别", category_options, default=category_options)
    else:
        selected_categories = []

    min_year = int(df["year"].min()) if "year" in df.columns and df["year"].notna().any() else None
    max_year = int(df["year"].max()) if "year" in df.columns and df["year"].notna().any() else None
    if min_year is not None and max_year is not None:
        selected_years = st.slider("年份范围", min_year, max_year, (min_year, max_year))
    else:
        selected_years = None

    country_options = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []
    selected_countries = st.multiselect("国家", country_options)

filtered_df = df.copy()
if selected_categories and "category_label" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["category_label"].isin(selected_categories)]
if selected_years and "year" in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df["year"] >= selected_years[0]) & (filtered_df["year"] <= selected_years[1])]
if selected_countries and "country" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]

st.subheader("1. 数据概况 KPI")
col1, col2, col3, col4, col5= st.columns(5)

time_span = "-"
if "year" in filtered_df.columns and filtered_df["year"].notna().any():
    time_span = f"{int(filtered_df['year'].min())}–{int(filtered_df['year'].max())}"

col1.metric("推文总数", format_number(len(filtered_df)))
col2.metric("时间跨度", time_span)
col3.metric("国家数量", format_number(filtered_df["country"].nunique()) if "country" in filtered_df.columns else "-")
col4.metric("用户数量", format_number(filtered_df["user_id"].nunique()) if "user_id" in filtered_df.columns else "-")
col5.metric("总互动量", format_number(filtered_df["engagement"].sum()) if "engagement" in filtered_df.columns else "-")

st.divider()
st.subheader("2. 数据类别构成")
if ensure_columns(filtered_df, ["category_label"]):
    category_count = (
        filtered_df.groupby("category_label", as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
        .sort_values("tweet_count", ascending=False)
    )
    fig_category = px.pie(
        category_count,
        names="category_label",
        values="tweet_count",
        hole=0.35,
        title="内容类别构成：美食与节庆文化",
        category_orders={"category_label": ["美食", "节庆文化"]},
    )
    fig_category.update_traces(textposition="inside", textinfo="percent+label")
    fig_category.update_layout(legend_title_text="内容类别")
    st.plotly_chart(fig_category, use_container_width=True)
else:
    add_empty_state("当前数据缺少 category 字段，无法展示类别构成。")

st.divider()
st.subheader("3. 时间趋势分析")
trend_tab1, trend_tab2, trend_tab3 = st.tabs(["年度趋势", "1–12月趋势", "月份热力图"])

with trend_tab1:
    if ensure_columns(filtered_df, ["year", "category_label"]):
        year_count = (
            filtered_df.dropna(subset=["year"])
            .groupby(["year", "category_label"], as_index=False)
            .size()
            .rename(columns={"size": "tweet_count"})
            .sort_values(["year", "category_label"])
        )
        fig_year = px.line(
            year_count,
            x="year",
            y="tweet_count",
            color="category_label",
            markers=True,
            title="年度推文数量变化：美食 vs 节庆文化",
            labels={"year": "年份", "tweet_count": "推文数", "category_label": "内容类别"},
            category_orders={"category_label": ["美食", "节庆文化"]},
        )
        fig_year.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig_year, use_container_width=True)

with trend_tab2:
    if ensure_columns(filtered_df, ["month", "category_label"]):
        month_count = (
            filtered_df.dropna(subset=["month"])
            .groupby(["month", "category_label"], as_index=False)
            .size()
            .rename(columns={"size": "tweet_count"})
        )
        all_months = pd.MultiIndex.from_product(
            [range(1, 13), category_order(filtered_df)], names=["month", "category_label"]
        ).to_frame(index=False)
        month_count = all_months.merge(month_count, on=["month", "category_label"], how="left").fillna({"tweet_count": 0})
        month_count["month_label"] = month_count["month"].astype(int).astype(str) + "月"

        fig_month = px.line(
            month_count,
            x="month",
            y="tweet_count",
            color="category_label",
            markers=True,
            title="1–12月推文数量变化：美食 vs 节庆文化",
            labels={"month": "月份", "tweet_count": "推文数", "category_label": "内容类别"},
            category_orders={"category_label": ["美食", "节庆文化"]},
        )
        fig_month.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=[f"{m}月" for m in range(1, 13)]))
        st.plotly_chart(fig_month, use_container_width=True)

with trend_tab3:
    if ensure_columns(filtered_df, ["year", "month", "category_label"]):
        heatmap_data = (
            filtered_df.dropna(subset=["year", "month"])
            .groupby(["category_label", "year", "month"], as_index=False)
            .size()
            .rename(columns={"size": "tweet_count"})
        )
        cats = category_order(filtered_df)
        if heatmap_data.empty:
            add_empty_state("筛选后没有可用于热力图的数据。")
        else:
            heat_cols = st.columns(len(cats)) if cats else []
            for idx, cat in enumerate(cats):
                cat_heat = heatmap_data[heatmap_data["category_label"] == cat]
                with heat_cols[idx]:
                    if cat_heat.empty:
                        add_empty_state(f"{cat} 没有可展示数据。")
                    else:
                        heatmap_pivot = cat_heat.pivot(index="year", columns="month", values="tweet_count").fillna(0)
                        heatmap_pivot = heatmap_pivot.reindex(columns=list(range(1, 13)), fill_value=0)
                        fig_heat = px.imshow(
                            heatmap_pivot,
                            labels=dict(x="月份", y="年份", color="推文数"),
                            x=[f"{m}月" for m in heatmap_pivot.columns],
                            y=heatmap_pivot.index.astype(int),
                            text_auto=True,
                            aspect="auto",
                            title=f"{cat}：年份 × 月份热力图",
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)

st.divider()
st.subheader("4. 空间分布分析")
if ensure_columns(filtered_df, ["country"]):
    country_top = (
        filtered_df.groupby("country", as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
        .sort_values("tweet_count", ascending=False)
        .head(10)
    )
    fig_country = px.bar(
        country_top.sort_values("tweet_count"),
        x="tweet_count",
        y="country",
        orientation="h",
        title="推文数 Top 10 国家",
        labels={"tweet_count": "推文数", "country": "国家"},
        text="tweet_count",
    )
    st.plotly_chart(fig_country, use_container_width=True)

        
if ensure_columns(filtered_df, ["country", "city"]):
    st.markdown("#### 国家选择与城市推文数量等值展示")
    available_countries = (
        filtered_df.groupby("country", as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
        .sort_values("tweet_count", ascending=False)
    )
    default_country = available_countries.iloc[0]["country"] if not available_countries.empty else None
    country_for_map = st.selectbox(
        "选择国家",
        available_countries["country"].tolist(),
        index=0 if default_country else None,
        help="选择一个国家后，下方显示该国家内部各城市的推文数量分布。",
    )

    selected_country_df = filtered_df[filtered_df["country"] == country_for_map].copy()
    city_in_country = (
        selected_country_df[selected_country_df["city"].notna() & (selected_country_df["city"] != "未知")]
        .groupby("city", as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
        .sort_values("tweet_count", ascending=False)
    )

    map_left, map_right = st.columns([1.15, 1])
    with map_left:
        country_map_df = pd.DataFrame(
            {
                "country_map": [normalize_country_name(country_for_map)],
                "tweet_count": [len(selected_country_df)],
                "country": [country_for_map],
            }
        )
        fig_selected_country = px.choropleth(
            country_map_df,
            locations="country_map",
            locationmode="country names",
            color="tweet_count",
            hover_name="country",
            title=f"{country_for_map} 推文数量地图",
            labels={"tweet_count": "推文数"},
        )
        fig_selected_country.update_geos(showframe=False, showcoastlines=True, projection_type="natural earth")
        st.plotly_chart(fig_selected_country, use_container_width=True)

    with map_right:
        if city_in_country.empty:
            add_empty_state(f"{country_for_map} 暂无可识别城市数据。")
        else:
            fig_city_area = px.treemap(
                city_in_country.head(30),
                path=["city"],
                values="tweet_count",
                color="tweet_count",
                title=f"{country_for_map} 城市推文数量等值展示",
                labels={"tweet_count": "推文数", "city": "城市"},
            )
            fig_city_area.update_traces(textinfo="label+value+percent parent")
            st.plotly_chart(fig_city_area, use_container_width=True)

   
st.divider()
st.subheader("5. 传播互动分析")
interaction_cols = [c for c in ["like_count", "rt_count", "reply_count", "quote_count"] if c in filtered_df.columns]
inter_col1, inter_col2 = st.columns(2)

with inter_col1:
    if interaction_cols:
        interaction_sum = pd.DataFrame(
            {
                "互动类型": [INTERACTION_LABELS.get(c, c) for c in interaction_cols],
                "互动量": [filtered_df[c].sum() for c in interaction_cols],
            }
        )
        interaction_sum = interaction_sum[interaction_sum["互动量"] > 0]
        if interaction_sum.empty:
            add_empty_state("筛选后互动量为 0，无法展示互动构成。")
        else:
            fig_interaction = px.pie(
                interaction_sum,
                names="互动类型",
                values="互动量",
                hole=0.35,
                title="互动量构成",
            )
            fig_interaction.update_traces(
                textposition="outside",
                textinfo="label+percent+value",
                hovertemplate="%{label}<br>互动量：%{value:,.0f}<br>占比：%{percent}<extra></extra>",
                automargin=True,
            )
            fig_interaction.update_layout(
                showlegend=True,
                legend_title_text="互动类型",
                margin=dict(t=70, b=70, l=70, r=70),
                uniformtext_minsize=11,
                uniformtext_mode="hide",
            )
            st.plotly_chart(fig_interaction, use_container_width=True)
    else:
        add_empty_state("缺少 rt_count / like_count / reply_count / quote_count 字段，无法展示互动构成。")

with inter_col2:
    if ensure_columns(filtered_df, ["category_label", "engagement"]):
        category_engagement = (
            filtered_df.groupby("category_label", as_index=False)["engagement"]
            .sum()
            .sort_values("engagement", ascending=False)
        )
        fig_cat_engage = px.bar(
            category_engagement,
            x="category_label",
            y="engagement",
            title="类别互动量对比：美食 vs 节庆文化",
            labels={"category_label": "内容类别", "engagement": "互动量"},
            text_auto=True,
            category_orders={"category_label": ["美食", "节庆文化"]},
        )
        st.plotly_chart(fig_cat_engage, use_container_width=True)

if ensure_columns(filtered_df, ["engagement"]):
    st.markdown("#### Top 10 高互动推文")
    display_cols = [
        col
        for col in [
            "category_label",
            "tweet_created_at",
            "country",
            "city",
            "text",
            "rt_count",
            "like_count",
            "reply_count",
            "quote_count",
            "engagement",
        ]
        if col in filtered_df.columns
    ]
    top_tweets = filtered_df.sort_values("engagement", ascending=False).head(10)[display_cols].copy()
    rename_map = {
        "category_label": "类别",
        "tweet_created_at": "发布时间",
        "country": "国家",
        "city": "城市",
        "text": "原文",
        "rt_count": "转发数",
        "like_count": "点赞数",
        "reply_count": "回复数",
        "quote_count": "引用数",
        "engagement": "互动量",
    }
    top_tweets = top_tweets.rename(columns=rename_map)
    st.dataframe(top_tweets, use_container_width=True, hide_index=True)

