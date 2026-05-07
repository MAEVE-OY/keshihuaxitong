# -*- coding: utf-8 -*-
"""
Streamlit 页面三：主题分析
主题分析：美食与节庆文化的主题结构及情感差异

运行方式：
    streamlit run 3_Topic_Analysis_3_final.py

数据文件：
    C:/bishe/xitong/data/processed/all_data.csv
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from wordcloud import STOPWORDS, WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False
    STOPWORDS = set()


# =========================
# 基础配置
# =========================
# 按要求固定使用该绝对路径，不设置其他回退路径。
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "all_data.csv"

st.set_page_config(
    page_title="主题分析：美食与节庆文化",
    page_icon="🏮",
    layout="wide",
)

CUSTOM_STOPWORDS = {
    "china", "chinese", "food", "festival", "new", "year", "day", "today",
    "one", "get", "good", "like", "would", "could", "also", "really",
    "http", "https", "co", "amp", "rt", "t", "s", "m", "u", "via",
    "people", "time", "make", "made", "go", "going", "got", "see", "think",
    "lunar", "happy", "celebrate", "celebration", "celebrating",
}

# 注意：这里不只排除完全等于 Unmapped Topic 的值，也排除 Unmapped Topic主题、Unmapped Topic 12、噪声/未分类主题等变体。
NOISE_TOPICS = {
    "", "nan", "none", "null", "noise", "outlier", "other", "others",
    "其它", "其他", "噪声", "未归类", "未分类", "未归类主题", "未分类主题", "噪声/未分类主题",
    "unmapped", "unmapped topic", "unmapped topic主题",
    "-1", "topic -1",
}

NOISE_TOPIC_PATTERNS = [
    r"^\s*unmapped\s*topic\b.*$",
    r"^\s*unmapped\b.*$",
    r"^\s*topic\s*-?1\b.*$",
    r"^\s*-1\s*$",
    r"^\s*noise\b.*$",
    r"^\s*outlier\b.*$",
    r"噪声",
    r"未分类",
    r"未归类",
    r"未归档",
]

REQUIRED_COLUMNS = [
    "category", "final_topic", "clean_text", "text", "year", "sentiment_label",
    "compound", "topic_prob", "country", "city", "engagement",
]

CATEGORY_LABELS = {
    "food": "Food 美食类",
    "festival": "Festival 节庆文化类",
    "all": "全部主题参照",
}


# =========================
# 数据读取与清洗
# =========================
@st.cache_data(show_spinner="正在读取主题分析数据...")
def load_data(path: str | Path) -> pd.DataFrame:
    """读取 CSV 并做基础类型处理。"""
    if not Path(path).exists():
        st.error(f"找不到数据文件：{path}")
        st.stop()

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # 兜底补齐缺失列，避免因为个别字段缺失导致页面崩溃
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].replace({"foods": "food", "festivals": "festival"})
    df["final_topic"] = df["final_topic"].astype(str).str.strip()
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    numeric_cols = [
        "year", "compound", "topic_prob", "engagement",
        "like_count", "rt_count", "reply_count", "quote_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = df["year"].astype("Int64")
    df["compound"] = df["compound"].fillna(0)
    df["topic_prob"] = df["topic_prob"].fillna(0)
    df["engagement"] = df["engagement"].fillna(0)

    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["text"] = df["text"].fillna(df["clean_text"]).astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    df["city"] = df["city"].fillna("").astype(str)

    return df


def normalize_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", str(topic).strip().lower())


def is_valid_topic(topic: str) -> bool:
    """判断是否为有效主题，排除噪声、未归类、Unmapped Topic 和空主题。"""
    t = normalize_topic(topic)
    if not t or t in NOISE_TOPICS:
        return False
    return not any(re.search(pattern, t, flags=re.IGNORECASE) for pattern in NOISE_TOPIC_PATTERNS)


def get_valid_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["final_topic"].map(is_valid_topic)].copy()


def get_topic_limit(valid_df: pd.DataFrame) -> int:
    """侧边栏主题数量：默认显示全部有效主题，而不是只显示 Top 10。"""
    topic_count = int(valid_df["final_topic"].nunique())
    if topic_count <= 0:
        return 1
    show_all = st.sidebar.checkbox("图表默认显示全部有效主题", value=True)
    if show_all:
        st.sidebar.caption(f"当前有效主题数：{topic_count} 个，图表默认全部展示。")
        return topic_count
    return st.sidebar.slider("图表显示主题数量", 1, topic_count, min(14, topic_count), 1)


def topic_count_label(s: pd.Series) -> Tuple[str, int]:
    if s.empty:
        return "暂无数据", 0
    counts = s.value_counts()
    return str(counts.index[0]), int(counts.iloc[0])


def top_mean_topic(df: pd.DataFrame, ascending: bool = False, min_count: int = 20) -> Tuple[str, float]:
    """按主题平均 compound 找最积极/最消极主题。"""
    if df.empty:
        return "暂无数据", 0.0
    agg = (
        df.groupby("final_topic")
        .agg(avg_compound=("compound", "mean"), n=("final_topic", "size"))
        .query("n >= @min_count")
        .sort_values("avg_compound", ascending=ascending)
    )
    if agg.empty:
        agg = (
            df.groupby("final_topic")
            .agg(avg_compound=("compound", "mean"), n=("final_topic", "size"))
            .sort_values("avg_compound", ascending=ascending)
        )
    row = agg.iloc[0]
    return str(agg.index[0]), float(row["avg_compound"])


def tokenize_text(texts: Iterable[str], extra_stopwords: Iterable[str] | None = None) -> List[str]:
    """简单英文词频分词，用于词云与主题高频词。"""
    stopwords = set(STOPWORDS) | CUSTOM_STOPWORDS
    if extra_stopwords:
        stopwords |= {str(w).lower() for w in extra_stopwords}

    joined = " ".join([str(t) for t in texts if pd.notna(t)]).lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z']{2,}", joined)
    return [w for w in tokens if w not in stopwords and len(w) > 2]


def make_wordcloud(texts: Iterable[str], title: str) -> None:
    """绘制词云。"""
    if not WORDCLOUD_AVAILABLE:
        st.warning("当前环境缺少 wordcloud 包。请先运行：pip install wordcloud")
        return

    tokens = tokenize_text(texts)
    if not tokens:
        st.info(f"{title} 暂无足够文本生成词云。")
        return

    text = " ".join(tokens)
    wc = WordCloud(
        width=900,
        height=420,
        background_color="white",
        max_words=120,
        collocations=False,
        random_state=42,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    st.pyplot(fig, clear_figure=True)


def horizontal_bar(data: pd.DataFrame, x: str, y: str, title: str, text_col: str | None = None):
    fig = px.bar(
        data,
        x=x,
        y=y,
        orientation="h",
        text=text_col,
        title=title,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(420, 32 * len(data) + 120),
        margin=dict(l=20, r=30, t=70, b=40),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def get_view_df(df: pd.DataFrame, category_key: str) -> pd.DataFrame:
    if category_key == "all":
        return df.copy()
    return df[df["category"] == category_key].copy()


def get_top_topics(df: pd.DataFrame, top_n: int) -> list[str]:
    return df["final_topic"].value_counts().head(top_n).index.tolist()


def render_sentiment_section(df: pd.DataFrame, category_key: str, top_n: int) -> None:
    """分 Food / Festival / 全部参照绘制主题情感分析。"""
    view_df = get_view_df(df, category_key)
    title_prefix = CATEGORY_LABELS.get(category_key, category_key)

    if view_df.empty:
        st.info(f"当前筛选条件下没有 {title_prefix} 数据。")
        return

    topic_list = get_top_topics(view_df, top_n)
    heat_df = view_df[view_df["final_topic"].isin(topic_list)].copy()

    st.markdown(f"#### {title_prefix}：主题 × 情感标签热力图")
    if not heat_df.empty:
        sentiment_order = [s for s in ["positive", "neutral", "negative"] if s in heat_df["sentiment_label"].unique()]
        other_sentiments = [s for s in sorted(heat_df["sentiment_label"].unique()) if s not in sentiment_order]
        sentiment_order += other_sentiments

        pivot = pd.crosstab(heat_df["final_topic"], heat_df["sentiment_label"], normalize="index")
        pivot = pivot.reindex(index=topic_list)
        pivot = pivot.reindex(columns=sentiment_order, fill_value=0)

        fig = px.imshow(
            pivot,
            text_auto=".1%",
            aspect="auto",
            title=f"{title_prefix}：主题 × 情感标签热力图（行内占比）",
            labels={"x": "情感标签", "y": "主题", "color": "占比"},
        )
        fig.update_layout(height=max(480, 36 * len(pivot) + 130))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无足够数据绘制情感结构热力图。")

    st.markdown(f"#### {title_prefix}：主题平均情感得分排序")
    avg_sent = (
        view_df.groupby("final_topic")
        .agg(avg_compound=("compound", "mean"), count=("final_topic", "size"))
        .reset_index()
    )
    avg_sent_show = avg_sent[avg_sent["final_topic"].isin(topic_list)].sort_values("avg_compound")
    if not avg_sent_show.empty:
        fig = px.bar(
            avg_sent_show,
            x="avg_compound",
            y="final_topic",
            orientation="h",
            hover_data=["count"],
            title=f"{title_prefix}：主题平均 compound 排序（右侧更积极，左侧更消极）",
            labels={"avg_compound": "平均 compound", "final_topic": "主题"},
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=max(440, 34 * len(avg_sent_show) + 140))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"#### {title_prefix}：主题情感强度分布")
    # 箱线图如果主题太多会很拥挤，默认展示当前视图下全部主题；超过 14 个时按数量取前 14 个。
    box_topic_list = get_top_topics(view_df, min(top_n, 14))
    box_df = view_df[view_df["final_topic"].isin(box_topic_list)].copy()
    if not box_df.empty:
        fig = px.box(
            box_df,
            x="compound",
            y="final_topic",
            orientation="h",
            points="outliers",
            title=f"{title_prefix}：compound 分布箱线图",
            labels={"compound": "compound 情感得分", "final_topic": "主题"},
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=max(460, 42 * len(box_topic_list) + 120))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无足够数据绘制箱线图。")


def render_time_section(df: pd.DataFrame, category_key: str, top_n: int) -> None:
    """分 Food / Festival / 全部参照绘制主题年度演化。"""
    view_df = get_view_df(df, category_key).dropna(subset=["year"]).copy()
    title_prefix = CATEGORY_LABELS.get(category_key, category_key)

    if view_df.empty:
        st.info(f"当前筛选条件下没有 {title_prefix} 年份数据。")
        return

    view_df["year"] = view_df["year"].astype(int)
    topic_list = get_top_topics(view_df, top_n)

    st.markdown(f"#### {title_prefix}：主要主题的年度演化")
    annual_counts = (
        view_df[view_df["final_topic"].isin(topic_list)]
        .groupby(["year", "final_topic"])
        .size()
        .reset_index(name="count")
    )
    yearly_total = view_df.groupby("year").size().rename("year_total").reset_index()
    annual_counts = annual_counts.merge(yearly_total, on="year", how="left")
    annual_counts["share"] = annual_counts["count"] / annual_counts["year_total"]

    annual_pivot = annual_counts.pivot_table(index="final_topic", columns="year", values="share", fill_value=0)
    annual_pivot = annual_pivot.reindex(index=topic_list)

    if not annual_pivot.empty:
        fig = px.imshow(
            annual_pivot,
            text_auto=".1%",
            aspect="auto",
            title=f"{title_prefix}：主题年度热力图（各主题在该类别当年推文中的占比）",
            labels={"x": "年份", "y": "主题", "color": "当年占比"},
        )
        fig.update_layout(height=max(500, 36 * len(annual_pivot) + 140))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无足够数据绘制年度热力图。")

    st.markdown(f"#### {title_prefix}：选定主题年度变化趋势")
    selectable_topics = sorted(view_df["final_topic"].dropna().unique())
    default_topics = topic_list[: min(3, len(topic_list))]
    selected_topics_for_line = st.multiselect(
        f"选择 1—3 个主题查看年度趋势（{title_prefix}）",
        options=selectable_topics,
        default=default_topics,
        max_selections=3,
        key=f"line_topics_{category_key}",
    )
    metric_mode = st.radio(
        f"趋势指标（{title_prefix}）",
        ["数量", "当年占比"],
        horizontal=True,
        key=f"metric_mode_{category_key}",
    )
    if selected_topics_for_line:
        line_counts = (
            view_df[view_df["final_topic"].isin(selected_topics_for_line)]
            .groupby(["year", "final_topic"])
            .size()
            .reset_index(name="count")
            .merge(yearly_total, on="year", how="left")
        )
        line_counts["share"] = line_counts["count"] / line_counts["year_total"]
        y_col = "count" if metric_mode == "数量" else "share"
        fig = px.line(
            line_counts,
            x="year",
            y=y_col,
            color="final_topic",
            markers=True,
            title=f"{title_prefix}：选定主题年度变化趋势",
            labels={"year": "年份", y_col: metric_mode, "final_topic": "主题"},
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# 页面标题与说明
# =========================
st.title("主题分析：美食与节庆文化的主题结构及情感差异")
st.caption(
    "本页面基于主题模型结果，对中国美食与节庆文化相关推文的核心讨论主题进行可视化展示，"
    "并结合情感分析结果，对比不同主题在内容结构、情绪倾向和时间演化上的差异。"
)

raw_df = load_data(DATA_PATH)
valid_df_all = get_valid_df(raw_df)
excluded_noise_rows = len(raw_df) - len(valid_df_all)

# =========================
# 侧边栏筛选
# =========================
st.sidebar.header("筛选条件")
st.sidebar.caption(f"读取数据路径：`{DATA_PATH}`")
st.sidebar.success(f"已排除噪声/未分类/Unmapped Topic：{excluded_noise_rows:,} 条")

available_categories = sorted([c for c in valid_df_all["category"].dropna().unique().tolist() if c in ["food", "festival"]])
selected_categories = st.sidebar.multiselect(
    "文化类别",
    options=available_categories,
    default=available_categories,
)

available_years = sorted([int(y) for y in valid_df_all["year"].dropna().unique().tolist()])
if available_years:
    year_range = st.sidebar.slider(
        "年份范围",
        min_value=min(available_years),
        max_value=max(available_years),
        value=(min(available_years), max(available_years)),
        step=1,
    )
else:
    year_range = (None, None)

available_sentiments = sorted([s for s in valid_df_all["sentiment_label"].dropna().unique().tolist() if s])
selected_sentiments = st.sidebar.multiselect(
    "情感标签",
    options=available_sentiments,
    default=available_sentiments,
)

min_topic_prob = st.sidebar.slider("最低主题置信度 topic_prob", 0.0, 1.0, 0.0, 0.05)
top_n = get_topic_limit(valid_df_all)

filtered_df = valid_df_all.copy()
if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
if available_years and year_range[0] is not None:
    filtered_df = filtered_df[(filtered_df["year"] >= year_range[0]) & (filtered_df["year"] <= year_range[1])]
if selected_sentiments:
    filtered_df = filtered_df[filtered_df["sentiment_label"].isin(selected_sentiments)]
filtered_df = filtered_df[filtered_df["topic_prob"] >= min_topic_prob]

if filtered_df.empty:
    st.warning("当前筛选条件下没有可展示的数据，请调整侧边栏筛选条件。")
    st.stop()

# =========================
# 1. KPI 概览
# =========================
st.header("1. 主题分析 KPI 概览")

valid_topic_count = filtered_df["final_topic"].nunique()
food_topic_count = filtered_df.loc[filtered_df["category"] == "food", "final_topic"].nunique()
festival_topic_count = filtered_df.loc[filtered_df["category"] == "festival", "final_topic"].nunique()
largest_topic, largest_topic_n = topic_count_label(filtered_df["final_topic"])
largest_food_topic, largest_food_n = topic_count_label(filtered_df.loc[filtered_df["category"] == "food", "final_topic"])
largest_festival_topic, largest_festival_n = topic_count_label(filtered_df.loc[filtered_df["category"] == "festival", "final_topic"])
most_positive_topic, most_positive_score = top_mean_topic(filtered_df, ascending=False)
most_negative_topic, most_negative_score = top_mean_topic(filtered_df, ascending=True)

def kpi_card(title, value, note=""):
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e6e6e6;
            border-radius: 14px;
            padding: 16px 18px;
            background-color: #ffffff;
            min-height: 145px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        ">
            <div style="
                font-size: 15px;
                color: #555;
                margin-bottom: 10px;
                font-weight: 600;
            ">
                {title}
            </div>
            <div style="
                font-size: 26px;
                line-height: 1.25;
                color: #262730;
                font-weight: 700;
                white-space: normal;
                word-break: break-word;
            ">
                {value}
            </div>
            <div style="
                font-size: 14px;
                color: #198754;
                margin-top: 10px;
                line-height: 1.3;
                white-space: normal;
                word-break: break-word;
            ">
                {note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


kpi_cols = st.columns(6)

with kpi_cols[0]:
    kpi_card("有效主题数", f"{valid_topic_count:,}", f"Food {food_topic_count} / Festival {festival_topic_count}")

with kpi_cols[1]:
    kpi_card("最大主题", largest_topic, f"{largest_topic_n:,} 条")

with kpi_cols[2]:
    kpi_card("最大美食主题", largest_food_topic, f"{largest_food_n:,} 条")

with kpi_cols[3]:
    kpi_card("最大节庆主题", largest_festival_topic, f"{largest_festival_n:,} 条")

with kpi_cols[4]:
    kpi_card("最积极主题", most_positive_topic, f"compound {most_positive_score:.3f}")

with kpi_cols[5]:
    kpi_card("最消极主题", most_negative_topic, f"compound {most_negative_score:.3f}")
st.info(
    "本页所有 KPI、图表、下拉框和表格均已排除噪声/未分类主题与 Unmapped Topic。"
    "默认展示全部有效主题；如果图太密，可在左侧取消“图表默认显示全部有效主题”后手动调节数量。"
)

st.divider()

# =========================
# 2. 美食与节庆主题对比
# =========================
st.header("2. 美食与节庆主题对比")
st.markdown("对比 Food 与 Festival 两类内容中最常出现的主题，回答“美食类和节庆类分别在谈什么”。")

left, right = st.columns(2)
for cat, col, label in [("food", left, "Food 美食类主题"), ("festival", right, "Festival 节庆文化类主题")]:
    with col:
        cat_df = filtered_df[filtered_df["category"] == cat]
        if cat_df.empty:
            st.info(f"当前筛选条件下没有 {cat} 数据。")
        else:
            cat_topic_n = int(cat_df["final_topic"].nunique())
            top_topics = (
                cat_df["final_topic"]
                .value_counts()
                .head(min(top_n, cat_topic_n))
                .rename_axis("final_topic")
                .reset_index(name="count")
            )
            fig = horizontal_bar(top_topics, "count", "final_topic", f"{label}（共 {cat_topic_n} 个有效主题）", "count")
            st.plotly_chart(fig, use_container_width=True)



st.divider()

# =========================
# 3. 词汇特征对比
# =========================
st.header("3. 词汇特征对比：Food 与 Festival 双词云")
wordcloud_mode = st.radio(
    "词云模式",
    ["按 Food / Festival 对比", "按类别内主题查看", "按任意主题查看"],
    horizontal=True,
)

if wordcloud_mode == "按 Food / Festival 对比":
    c1, c2 = st.columns(2)
    with c1:
        make_wordcloud(filtered_df.loc[filtered_df["category"] == "food", "clean_text"], "Food 美食类关键词词云")
    with c2:
        make_wordcloud(filtered_df.loc[filtered_df["category"] == "festival", "clean_text"], "Festival 节庆文化类关键词词云")
elif wordcloud_mode == "按类别内主题查看":
    c1, c2 = st.columns(2)
    with c1:
        wc_cat = st.selectbox("选择类别", ["food", "festival"], format_func=lambda x: CATEGORY_LABELS[x], key="wc_cat")
    cat_df_for_wc = filtered_df[filtered_df["category"] == wc_cat]
    with c2:
        topic_for_wc = st.selectbox("选择该类别下的主题", sorted(cat_df_for_wc["final_topic"].dropna().unique()), key="wc_topic_cat")
    make_wordcloud(cat_df_for_wc.loc[cat_df_for_wc["final_topic"] == topic_for_wc, "clean_text"], f"{CATEGORY_LABELS[wc_cat]}：{topic_for_wc} 关键词词云")
else:
    topic_for_wc = st.selectbox("选择主题生成词云", sorted(filtered_df["final_topic"].dropna().unique()), key="wc_topic_all")
    make_wordcloud(filtered_df.loc[filtered_df["final_topic"] == topic_for_wc, "clean_text"], f"{topic_for_wc} 关键词词云")

st.divider()

# =========================
# 4. 主题与情感融合分析
# =========================
st.header("4. 主题与情感融合分析")
st.markdown(
    "为避免美食主题和节庆主题混在一起造成解释困难，本模块按 Food、Festival 和全部参照分开展示。"
)
food_tab, festival_tab, all_tab = st.tabs(["Food 美食主题", "Festival 节庆文化主题", "全部主题参照"])
with food_tab:
    render_sentiment_section(filtered_df, "food", top_n)
with festival_tab:
    render_sentiment_section(filtered_df, "festival", top_n)
with all_tab:
    render_sentiment_section(filtered_df, "all", top_n)

st.divider()

# =========================
# 5. 主题时间演化
# =========================
st.header("5. 主题时间演化")
st.markdown(
    "这里不重复总体推文趋势，而是观察不同主题在各年份中的突出程度。"
    "年度占比分母在 Food/Festival 标签页中分别使用该类别当年的推文总数，因此更适合做类别内部比较。"
)
food_time_tab, festival_time_tab, all_time_tab = st.tabs(["Food 美食主题", "Festival 节庆文化主题", "全部主题参照"])
with food_time_tab:
    render_time_section(filtered_df, "food", top_n)
with festival_time_tab:
    render_time_section(filtered_df, "festival", top_n)
with all_time_tab:
    render_time_section(filtered_df, "all", top_n)

st.divider()

# =========================
# 6. 主题详情与代表文本
# =========================
st.header("6. 主题详情与代表文本")

detail_cat = st.radio(
    "先选择查看范围",
    ["food", "festival", "all"],
    format_func=lambda x: CATEGORY_LABELS[x],
    horizontal=True,
    key="detail_cat",
)
detail_df = get_view_df(filtered_df, detail_cat)

if detail_df.empty:
    st.info("当前范围暂无数据。")
    st.stop()

selected_topic = st.selectbox("选择主题查看详情", sorted(detail_df["final_topic"].dropna().unique()))
topic_df = detail_df[detail_df["final_topic"] == selected_topic].copy()

if topic_df.empty:
    st.info("当前主题暂无数据。")
    st.stop()

st.subheader("主题详情卡片")
main_category = topic_df["category"].mode().iloc[0] if not topic_df["category"].mode().empty else "未知"
avg_compound = topic_df["compound"].mean()
avg_prob = topic_df["topic_prob"].mean()
text_count = len(topic_df)
sent_dist = topic_df["sentiment_label"].value_counts(normalize=True).rename("share").reset_index()
sent_dist.columns = ["sentiment_label", "share"]

def detail_card(title, value):
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e6e6e6;
            border-radius: 14px;
            padding: 16px 18px;
            background-color: #ffffff;
            min-height: 125px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        ">
            <div style="
                font-size: 15px;
                color: #555;
                margin-bottom: 10px;
                font-weight: 600;
            ">
                {title}
            </div>
            <div style="
                font-size: 24px;
                line-height: 1.3;
                color: #262730;
                font-weight: 700;
                white-space: normal;
                word-break: break-word;
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


row1 = st.columns([2, 1, 1])
row2 = st.columns(2)

with row1[0]:
    detail_card("主题名称", selected_topic)

with row1[1]:
    detail_card("推文数量", f"{text_count:,}")

with row1[2]:
    detail_card("主要类别", str(main_category).title())

with row2[0]:
    detail_card("平均情感得分", f"{avg_compound:.3f}")

with row2[1]:
    detail_card("平均主题置信度", f"{avg_prob:.3f}")

c1, c2 = st.columns([1, 1])
with c1:
    if not sent_dist.empty:
        fig = px.pie(
            sent_dist,
            names="sentiment_label",
            values="share",
            title="该主题内部情感比例",
        )
        st.plotly_chart(fig, use_container_width=True)
with c2:
    topic_tokens = tokenize_text(topic_df["clean_text"])
    top_words = Counter(topic_tokens).most_common(10)
    if top_words:
        words_df = pd.DataFrame(top_words, columns=["keyword", "count"])
        fig = px.bar(words_df.sort_values("count"), x="count", y="keyword", orientation="h", title="该主题高频词 Top 10")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("该主题暂无足够文本提取高频词。")

st.subheader("选定主题的代表性推文")
sentiment_filter = st.selectbox("选择情感类型", ["all"] + sorted(topic_df["sentiment_label"].dropna().unique().tolist()))
rep_df = topic_df.copy()
if sentiment_filter != "all":
    rep_df = rep_df[rep_df["sentiment_label"] == sentiment_filter]

if sentiment_filter == "positive":
    rep_df = rep_df.sort_values(["compound", "topic_prob"], ascending=[False, False])
elif sentiment_filter == "negative":
    rep_df = rep_df.sort_values(["compound", "topic_prob"], ascending=[True, False])
else:
    rep_df = rep_df.sort_values(["topic_prob", "engagement"], ascending=[False, False])

show_cols = [
    "text", "category", "sentiment_label", "compound", "topic_prob",
    "country", "city", "engagement",
]
show_cols = [c for c in show_cols if c in rep_df.columns]
st.dataframe(
    rep_df[show_cols].head(100),
    use_container_width=True,
    hide_index=True,
)

csv = rep_df[show_cols].to_csv(index=False).encode("utf-8-sig")
safe_topic_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", selected_topic)[:80]
st.download_button(
    "下载当前主题代表性推文 CSV",
    data=csv,
    file_name=f"representative_tweets_{safe_topic_name}.csv",
    mime="text/csv",
)

st.caption(
    "说明：本页重点展示主题结构、主题情感差异、主题年度演化和代表文本，"
    "避免重复总览页中的总体传播规模、空间分布和整体情感占比。"
)