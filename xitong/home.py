import streamlit as st

# 1. 设置页面
st.set_page_config(
    page_title="首页 | 中国美食与节庆文化分析系统",
    page_icon="🏮",
    layout="wide",
)

# 2. 定义 CSS 样式 (去掉 dedent，保持左对齐)
main_style = """
<style>
    .main .block-container { padding-top: 1.4rem; max-width: 1280px; }
    .stApp {
        background: radial-gradient(circle at 8% 12%, rgba(180, 35, 24, 0.08), transparent 26%),
                    linear-gradient(135deg, #FBF4E8 0%, #F7EAD8 48%, #FDF8EF 100%);
    }
    .cover {
        min-height: 80vh;
        border-radius: 34px;
        padding: 2.5rem;
        background: linear-gradient(135deg, rgba(255, 252, 244, 0.95), rgba(250, 238, 214, 0.9));
        border: 1px solid rgba(143, 52, 38, 0.18);
        box-shadow: 0 20px 50px rgba(88, 39, 20, 0.15);
        position: relative;
    }
    .title { font-size: 3rem; font-weight: 900; color: #5F1F18; line-height: 1.2; margin: 1rem 0; }
    .subtitle { color: #6B3F2A; line-height: 1.8; max-width: 800px; }
    .module-row { display: flex; gap: 1.5rem; margin-top: 2rem; }
    .module { 
        flex: 1; background: white; padding: 1.5rem; border-radius: 20px;
        border: 1px solid rgba(154, 52, 18, 0.1);
    }
    .module-title { font-weight: bold; color: #7C2D12; margin-bottom: 0.5rem; }
</style>
"""

# 3. 定义 HTML 内容
main_html = """
<div class="cover">
    <div style="font-size: 2.5rem;">🏮</div>
    <div style="color: #9A3412; font-weight: 800; letter-spacing: 0.2em; font-size: 0.8rem;">CHINESE CULTURE ANALYSIS</div>
    <div class="title">中国美食与节庆文化<br>社交媒体传播数据分析系统</div>
    <div style="width: 100px; height: 4px; background: #A83226; margin: 1.5rem 0;"></div>
    <div class="subtitle">
        本系统以中国美食文化与节庆文化相关社交媒体文本为研究对象，展示传播概况、情感倾向与主题结构。
    </div>
    <div class="module-row">
        <div class="module">
            <div class="module-title">一、数据总览</div>
            <div style="font-size: 0.9rem; color: #666;">展示推文规模、时间趋势、地理分布。</div>
        </div>
        <div class="module">
            <div class="module-title">二、情感分析</div>
            <div style="font-size: 0.9rem; color: #666;">对比美食与节庆的正负情绪差异。</div>
        </div>
        <div class="module">
            <div class="module-title">三、主题分析</div>
            <div style="font-size: 0.9rem; color: #666;">挖掘核心话题与关键词特征。</div>
        </div>
    </div>
</div>
"""

# 4. 统一渲染 (关键：确保 unsafe_allow_html=True)
st.markdown(main_style, unsafe_allow_html=True)
st.markdown(main_html, unsafe_allow_html=True)