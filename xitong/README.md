# Xitong Streamlit App
一个多页面的 Streamlit 应用，用于数据概览、情感分析和主题分析。
xitong/
├── home.py # 主页面入口
├── pages/ # 多页面子模块
│ ├── 1_Overview_4.py # 数据概览页面
│ ├── 2_Sentiment_Analysis_2.py # 情感分析页面
│ └── 3_Topic_Analysis_3.py # 主题分析页面
├── data/
│ ├── processed/ # 处理后的数据
│ │ ├── all_data.csv
│ │ ├── all_topic_mapping.csv
│ │ └── all_topic_summary.csv
│ └── raw/ # 原始数据
│ ├── festival_summary_with_final_topics_2.csv
│ ├── festival_topic_mapping_2.csv
│ ├── festival_vader1.csv
│ ├── festival_with_final_topics_2.csv
│ ├── food_summary_with_final_topics_2.csv
│ ├── food_topic_mapping_2.csv
│ ├── food_vader1.csv
│ └── food_with_final_topics_2.csv
└── requirements.txt # Python依赖文件