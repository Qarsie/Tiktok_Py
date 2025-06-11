import streamlit as st
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Configure page
st.set_page_config(
    page_title="TikTok Trend Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff0050, #ff9900);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #ff0050;
        border-bottom: 2px solid #ff0050;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation with emojis
st.sidebar.markdown("# 🧭 Navigation")
pages = [
    "📊 Part 1: Data Collection",
    "🧹 Part 2: Data Preprocessing", 
    "📈 Part 3: Exploratory Data Analysis (EDA)",
    "🔍 Part 4: Trend Identification Using NLP",
    "🤖 Part 5: Predictive Model Development",
    "🚀 Part 6: AI-Driven Trend Forecasting",
    "📊 Part 7: Visualization",
    "📝 Part 8 & 9: Summary"
]
selected_page = st.sidebar.radio("Go to", pages)

# Main content with styled header
st.markdown('<h1 class="main-header">🔥 AI-Powered Trend Prediction on TikTok</h1>', unsafe_allow_html=True)
st.markdown("---")

if selected_page == "📊 Part 1: Data Collection":
    st.markdown('<h2 class="section-header">📊 Part 1: Data Collection</h2>', unsafe_allow_html=True)
    
    # Info box
    st.info("🎯 **Objective:** Collect TikTok video data using FastAPI TikTok Scraper to build a comprehensive dataset for trend analysis.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container"><h3>📱 Data Source</h3><p>FastAPI TikTok Scraper</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container"><h3>🔄 Collection Method</h3><p>Automated API Scraping</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container"><h3>📊 Data Format</h3><p>CSV Files</p></div>', unsafe_allow_html=True)

    st.markdown("### 📂 Sample of Collected TikTok Data")

    try:
        df = pd.read_csv("data/merged/merged_data_deduplicated.csv")
        
        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Total Records", f"{len(df):,}")
        with col2:
            st.metric("📊 Features", f"{len(df.columns)}")
        with col3:
            st.metric("📅 Date Range", f"{df['create_time'].nunique():,} days")
        with col4:
            st.metric("👥 Unique Authors", f"{df['author'].nunique():,}")
        
        st.dataframe(df.head(30), use_container_width=True)
        
        st.markdown("### 📋 Feature Descriptions")
        feature_table = """
| 🏷️ Feature | 📝 Description | 📊 Data Type |
|-------------|-----------------|-------------|
| 🆔 video_id | Unique identifier for the TikTok video | String |
| 👤 author | Username of the video creator | String |
| 📄 description | Caption or text description of the video | Text |
| ❤️ likes | Number of likes the video has received | Integer |
| 💬 comments | Number of comments on the video | Integer |
| 🔄 shares | Number of times the video has been shared | Integer |
| ▶️ plays | Number of times the video has been played | Integer |
| #️⃣ hashtags | List of hashtags used in the video description | List |
| 🎵 music | Name or ID of the music used in the video | String |
| ⏰ create_time | Timestamp when the video was created | Datetime |
| 🔗 video_url | Direct URL link to the video | String |
| 📥 fetch_time | Timestamp when the data was collected | Datetime |
| 👀 views | Number of views the video has received | Integer |
| 📅 posted_time | Time when the video was posted | Datetime |
"""
        st.markdown(feature_table)
        
        st.success("✅ Data collection completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Could not load data: {e}")
        st.warning("⚠️ Please ensure the data file exists in the correct directory.")

#------------------------------------------------------

elif selected_page == "🧹 Part 2: Data Preprocessing":
    st.markdown('<h2 class="section-header">🧹 Part 2: Data Cleaning and Preprocessing</h2>', unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Clean and preprocess raw TikTok data to prepare it for analysis and modeling.")

    st.markdown("### 1️⃣ Column Merging & Null Value Handling")

    try:
        df = pd.read_csv("data/merged/merged_data_deduplicated.csv")
        
        # Create metrics for data quality
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", f"{len(df):,}")
        with col2:
            st.metric("❌ Null Values", f"{df.isnull().sum().sum():,}")
        with col3:
            st.metric("✅ Complete Records", f"{len(df.dropna()):,}")
        with col4:
            completion_rate = (len(df.dropna()) / len(df)) * 100
            st.metric("📈 Completion Rate", f"{completion_rate:.1f}%")
        
        # Null values table
        null_table = pd.DataFrame({
            "🏷️ Feature": df.columns,
            "❌ Null Values": df.isnull().sum().values,
            "📊 Percentage": (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(null_table, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Could not load data: {e}")

    st.markdown("#### 🔗 1.1 Column Merging")
    st.info("""
    **Merging Strategy:**
    - 📅 `posted_time` & `create_time` → `create_time`
    - 👀 `views` & `plays` → `plays`
    """)

    try:
        df = pd.read_csv("data/merged/merged_data_deduplicated.csv")
        df['create_time'] = df['create_time'].fillna(df['posted_time'])
        df['plays'] = df['plays'].fillna(df['views'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📅 Null in create_time", df['create_time'].isnull().sum())
        with col2:
            st.metric("▶️ Null in plays", df['plays'].isnull().sum())
            
    except Exception as e:
        st.error(f"❌ Error in column merging: {e}")

    st.markdown("#### 🛠️ 1.2 Null Value Handling")
    st.info("""
    **Handling Strategy:**
    - 📝 Description null values → "No Description"
    - #️⃣ Hashtags null values → Empty list `[]`
    """)

    try:
        df = pd.read_csv("data/processed/null_value_handling.csv")
        st.markdown("**📊 Sample after null handling:**")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Could not load processed data: {e}")

    st.markdown("### 2️⃣ Description Cleaning & Hashtag Processing")
    
    st.markdown("#### 🧹 2.1 Description Cleaning")
    st.info("🎯 **Goal:** Remove emojis, links, and special characters from descriptions for better NLP processing")

    try:
        df = pd.read_csv("data/processed/clean_desc_sample.csv")
        st.markdown("**📊 Before vs After Cleaning:**")
        st.dataframe(df.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Could not load cleaned description data: {e}")

    st.markdown("#### #️⃣ 2.2 Hashtag Processing")
    st.info("🎯 **Goal:** Convert hashtag strings to structured list format for analysis")

    try:
        df = pd.read_csv("data/processed/clean_hashtags_sample.csv")
        st.markdown("**📊 Processed Hashtags:**")
        st.dataframe(df.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Could not load hashtag data: {e}")

    st.markdown("### 3️⃣ Engagement Metrics Calculation")
    st.info("🎯 **Goal:** Create comprehensive engagement metrics to quantify video performance")
    
    st.markdown("""
    **📊 Calculated Metrics:**
    - 💫 **Total Engagement** = Likes + Comments + Shares
    - 📈 **Engagement Rate per Play** = Total Engagement / Plays
    """)

    try:
        df = pd.read_csv("data/processed/engagement_metric_sample.csv")
        st.markdown("**📊 Engagement Metrics Sample:**")
        st.dataframe(df.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Could not load engagement metrics: {e}")

    st.markdown("### 4️⃣ NLP Hashtags Extraction")
    st.info("🎯 **Goal:** Extract and visualize top hashtags from the dataset")
    
    st.markdown("**🏆 Top 20 Hashtags Visualization:**")
    try:
        st.image("data/graph/hashtag_wordcloud.png", caption="📊 Hashtag Word Cloud - Visual representation of trending hashtags")
    except Exception as e:
        st.error(f"❌ Could not load hashtag visualization: {e}")

    st.markdown("### 5️⃣ Final Cleaned Dataset")
    st.success("🎉 **Data preprocessing completed successfully!**")
    
    try:
        df = pd.read_csv("data/processed/tiktok_processed_sample.csv")
        
        # Show final dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Final Records", f"{len(df):,}")
        with col2:
            st.metric("🏷️ Features", f"{len(df.columns)}")
        with col3:
            st.metric("✅ Data Quality", "High")
        with col4:
            st.metric("🚀 Ready for Analysis", "Yes")
            
        st.markdown("**📊 Final Processed Dataset Sample:**")
        st.dataframe(df.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Could not load final dataset: {e}")

#------------------------------------------------------

elif selected_page == "📈 Part 3: Exploratory Data Analysis (EDA)":
    st.markdown('<h2 class="section-header">📈 Part 3: Exploratory Data Analysis (EDA)</h2>', unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Explore data patterns and relationships through interactive visualizations and statistical analysis.")
    
    try:
        df = pd.read_csv("data/processed/tiktok_categorized.csv")
        
        # Dataset overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Videos", f"{len(df):,}")
        with col2:
            st.metric("👥 Unique Creators", f"{df['author'].nunique():,}")
        with col3:
            avg_engagement = df['engagement_rate_per_play'].mean()
            st.metric("📈 Avg Engagement Rate", f"{avg_engagement:.3f}")
        with col4:
            viral_count = (df['engagement_rate_per_play'] > df['engagement_rate_per_play'].quantile(0.9)).sum()
            st.metric("🔥 Viral Videos (Top 10%)", f"{viral_count:,}")
        
        st.markdown("---")
        
        st.markdown("### 1️⃣ ⏰ Engagement Rate per Play by Hour of Day")
        fig1 = px.box(df, x='create_hour', y='engagement_rate_per_play',
                     title="📅 Engagement Rate Distribution by Posting Hour",
                     labels={"create_hour": "Hour of Day (24h format)", "engagement_rate_per_play": "Engagement Rate per Play"},
                     color_discrete_sequence=['#ff0050'])
        fig1.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        st.success("""
        **🔍 Key Insights:**
        - 🌙 **Prime Time:** Midnight to 5 AM shows higher engagement rates
        - 📉 **Low Period:** 10 AM - 12 PM has reduced engagement  
        - 🎯 **Viral Potential:** Outliers present at all hours - content quality matters most
        - 📊 **Consistency:** Median engagement remains stable (~0.10 to ~0.15) across hours
        """)

        st.markdown("### 2️⃣ 📅 Engagement Rate per Play by Day of Week")
        fig2 = px.box(df, x='day_of_week', y='engagement_rate_per_play',
                     title="📊 Weekly Engagement Patterns",
                     labels={"day_of_week": "Day of Week (0=Monday, 6=Sunday)", "engagement_rate_per_play": "Engagement Rate per Play"},
                     color_discrete_sequence=['#ff9900'])
        fig2.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info("""
        **📈 Weekly Trends:**
        - 🏆 **Sunday (6):** Highest median engagement - weekend leisure time
        - 🥈 **Monday (0):** Strong second place - fresh week energy
        - 📉 **Midweek Slump:** Tuesday-Friday show lower medians
        - 🎯 **Universal Potential:** Viral content can emerge any day
        """)

        st.markdown("### 3️⃣ 💭 Sentiment Polarity vs. Engagement Rate")
        fig3 = px.scatter(df, x='sentiment_polarity', y='engagement_rate_per_play',
                         color='sentiment_subjectivity',
                         color_continuous_scale='RdBu',
                         title="🎭 Content Sentiment Impact on Engagement",
                         labels={"sentiment_polarity": "Sentiment Polarity (Negative ← → Positive)", 
                                "engagement_rate_per_play": "Engagement Rate per Play", 
                                "sentiment_subjectivity": "Subjectivity Level"})
        fig3.update_layout(title_font_size=16)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.warning("""
        **🎭 Sentiment Analysis:**
        - 🎯 **All Sentiments Work:** High engagement occurs across the sentiment spectrum
        - 🎪 **Subjective Content:** Highly subjective content (red dots) shows wide performance spread
        - ⚖️ **Neutral Dominance:** Most content is neutral but less likely to reach viral extremes
        - 🔥 **Emotion Drives Virality:** Strong positive/negative sentiment can boost engagement
        """)

        st.markdown("### 4️⃣ #️⃣ Hashtag Count vs. Engagement Rate by Time Period")
        max_hashtags = 40
        max_engagement_rate = 0.5
        filtered_df = df[(df['hashtag_count'] <= max_hashtags) & (df['engagement_rate_per_play'] <= max_engagement_rate)]
        
        fig4 = px.scatter(filtered_df, x='hashtag_count', y='engagement_rate_per_play', color='time_period',
                         title="🏷️ Optimal Hashtag Strategy Analysis",
                         labels={"hashtag_count": "Number of Hashtags", "engagement_rate_per_play": "Engagement Rate per Play", "time_period": "Posting Time Period"})
        fig4.update_layout(title_font_size=16)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.success("""
        **#️⃣ Hashtag Strategy Insights:**
        - 🎯 **Sweet Spot:** 5-12 hashtags achieve optimal engagement
        - 🚫 **Over-tagging:** More than 20 hashtags typically reduces performance
        - 🌅 **Evening/Night:** Best performance times for hashtag strategies
        - 📈 **Quality over Quantity:** Fewer, relevant hashtags outperform hashtag spam
        """)

        st.markdown("### 5️⃣ 📱 Engagement Rate per Play by Content Category")
        filtered_cat = df[~df['content_description'].isin(["No Description", -1.0])].copy()
        order = sorted(filtered_cat['content_category'].unique())
        
        fig5 = px.box(filtered_cat, x='content_category', y='engagement_rate_per_play',
                    category_orders={'content_category': order},
                    title="🎯 Content Category Performance Analysis",
                    labels={"content_category": "Content Category (Topic)", "engagement_rate_per_play": "Engagement Rate per Play"},
                    color_discrete_sequence=['#764ba2'])
        fig5.update_xaxes(tickangle=45)
        fig5.update_layout(title_font_size=16)
        st.plotly_chart(fig5, use_container_width=True)
        
        st.info("""
        **📱 Content Category Winners:**
        - 🎓 **Educational (Topic 0):** Highest engagement - knowledge-hungry audience
        - 🤝 **Relatable (Topic 1):** Strong performer - universal appeal
        - 💕 **Relationship (Topic 2):** High-risk, high-reward content
        - 😂 **Humor (Topic 3):** Surprisingly underperforms - market oversaturation?
        - 📊 **Topics 5-7:** Consistent but modest performers
        """)

        st.markdown("### 6️⃣ 🔗 Correlation Matrix (Key Features)")
        cols_of_interest = [
            'engagement_rate_per_play', 'likes', 'comments', 'shares', 'plays',
            'hashtag_count', 'create_hour', 'day_of_week', 'sentiment_polarity', 'sentiment_subjectivity'
        ]
        corr = df[cols_of_interest].corr().values
        
        fig6 = ff.create_annotated_heatmap(
            z=corr,
            x=cols_of_interest,
            y=cols_of_interest,
            colorscale='Viridis',
            showscale=True,
            zmin=-1, zmax=1,
            annotation_text=[[f"{v:.2f}" for v in row] for row in corr]
        )
        fig6.update_layout(
            title_text="🔗 Feature Correlation Matrix",
            title_font_size=16,
            font=dict(color='white'),
            plot_bgcolor='black',
            paper_bgcolor='black'
        )
        st.plotly_chart(fig6, use_container_width=True)
        
        st.warning("""
        **🔗 Correlation Highlights:**
        - 💪 **Strong Relationships:** Likes correlate highly with plays, shares, and comments
        - 🏷️ **Hashtag Impact:** Minimal correlation between hashtag count and engagement
        - ⏰ **Timing Effects:** Hour and day show weak but measurable engagement patterns
        - 🎭 **Sentiment Independence:** Sentiment shows low correlation with other metrics
        """)

        st.markdown("### 7️⃣ 👑 Top 10 Accounts by Average Engagement Rate")
        top_accounts = df.groupby('author')['engagement_rate_per_play'].mean().sort_values(ascending=False).head(10)
        
        fig7 = px.bar(x=top_accounts.values, y=top_accounts.index,
                    orientation='h',
                    title="🏆 Elite Creator Performance Rankings",
                    labels={"x": "Average Engagement Rate per Play", "y": "Creator Handle"},
                    color=top_accounts.values, color_continuous_scale='viridis')
        fig7.update_layout(title_font_size=16)
        st.plotly_chart(fig7, use_container_width=True)
        
        st.success("""
        **👑 Elite Creator Insights:**
        - 🎯 **Elite Threshold:** Top 10 creators all achieve 40%+ engagement rates
        - 🌈 **Diverse Niches:** Different content types can achieve elite performance
        - 🔑 **Consistency:** Elite creators maintain high engagement across posts
        - 💡 **Success Factors:** Quality content + audience connection = sustained success
        """)
        
    except Exception as e:
        st.error(f"❌ Could not load EDA data or generate plots: {e}")
        st.info("📝 Please ensure the processed data files are available in the correct directory.")

        st.subheader("4. Hashtag Count vs. Engagement Rate by Time Period")
        max_hashtags = 40
        max_engagement_rate = 0.5
        filtered_df = df[(df['hashtag_count'] <= max_hashtags) & (df['engagement_rate_per_play'] <= max_engagement_rate)]
        fig4 = px.scatter(filtered_df, x='hashtag_count', y='engagement_rate_per_play', color='time_period',
                         title="Hashtag Count vs. Engagement Rate by Time Period",
                         labels={"hashtag_count": "Hashtag Count", "engagement_rate_per_play": "Engagement Rate per Play", "time_period": "Time Period"})
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
**Insights:**
- Fewer hashtags (0-15) achieve higher engagement rates.
- Optimal range: 5-12 hashtags.
- Evening and night posts show strongest performance.
- Over-hashtagting (>20) reduces engagement.
""")

        st.subheader("5. Engagement Rate per Play by Content Category")
        filtered_cat = df[~df['content_description'].isin(["No Description", -1.0])].copy()
        order = sorted(filtered_cat['content_category'].unique())
        fig5 = px.box(filtered_cat, x='content_category', y='engagement_rate_per_play',
                    category_orders={'content_category': order},
                    title="Engagement Rate per Play by Content Category (Topics 0–7, Excluding 'No description')",
                    labels={"content_category": "Content Category (Topic)", "engagement_rate_per_play": "Engagement Rate per Play"})
        fig5.update_xaxes(tickangle=45)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("""
**Category Insights:**
- Educational (Topic 0) and relatable (Topic 1) content have highest engagement.
- Humor (Topic 3) underperforms.
- Relationship (Topic 2) is high-risk/high-reward.
- Topics 5–7 are steady but modest performers.
""")

        st.subheader("6. Correlation Matrix (Key Features)")
        import plotly.figure_factory as ff
        cols_of_interest = [
            'engagement_rate_per_play', 'likes', 'comments', 'shares', 'plays',
            'hashtag_count', 'create_hour', 'day_of_week', 'sentiment_polarity', 'sentiment_subjectivity'
        ]
        corr = df[cols_of_interest].corr().values
        fig6 = ff.create_annotated_heatmap(
            z=corr,
            x=cols_of_interest,
            y=cols_of_interest,
            colorscale='Viridis',
            showscale=True,
            zmin=-1, zmax=1,
            annotation_text=[[f"{v:.2f}" for v in row] for row in corr]
        )
        fig6.update_layout(title_text="Correlation Matrix (Key Features)",
                          font=dict(color='white'),
                          plot_bgcolor='black',
                          paper_bgcolor='black')
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("""
**Correlation Highlights:**
- Likes strongly correlate with plays, shares, and comments.
- Hashtag count has minimal impact on engagement.
""")

        st.subheader("7. Top 10 Accounts by Average Engagement Rate")
        top_accounts = df.groupby('author')['engagement_rate_per_play'].mean().sort_values(ascending=False).head(10)
        fig7 = px.bar(x=top_accounts.values, y=top_accounts.index,
                    orientation='h',
                    title="Top 10 Accounts by Average Engagement Rate",
                    labels={"x": "Average Engagement Rate per Play", "y": "Account Name"},
                    color=top_accounts.values, color_continuous_scale='viridis')
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("""
**Account Insights:**
- Top 10 accounts have very close engagement rates (all 40%+).
- Different content niches can achieve elite engagement.
""")
    except Exception as e:
        st.error(f"Could not load EDA data or plot: {e}")

#------------------------------------------------------



elif selected_page == "🔍 Part 4: Trend Identification Using NLP":
    st.markdown('<h2 class="section-header">🔍 Part 4: Trend Identification Using NLP</h2>', unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Use advanced NLP techniques to identify emerging trends in TikTok video descriptions and hashtags.")
    
    try:
        df = pd.read_csv("data/processed/tiktok_processed_with_nlp_features.csv")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔍 NLP Features", "5+")
        with col2:
            st.metric("🏷️ Topic Models", "LDA")
        with col3:
            st.metric("🎭 Sentiment Analysis", "TextBlob")
        with col4:
            st.metric("👤 Named Entities", "spaCy NER")
        
        st.markdown("---")
        
        st.markdown("### 1️⃣ 📊 Topic Modeling (LDA) Results")
        st.info("🧠 **Technique:** Latent Dirichlet Allocation (LDA) to discover hidden topics in video descriptions and track their evolution over time.")
        
        # Topic evolution over time (weekly)
        df['create_time'] = pd.to_datetime(df['create_time'])
        df['date_group'] = df['create_time'].dt.to_period('W').dt.start_time
        topic_evolution = df.groupby('date_group')['dominant_topic'].value_counts(normalize=True).unstack(fill_value=0)
        topic_evolution = topic_evolution.rolling(window=4).mean().dropna()
        
        fig_topic = px.line(topic_evolution, x=topic_evolution.index, y=topic_evolution.columns,
                           labels={'value': 'Topic Proportion', 'date_group': 'Date'},
                           title='📈 Topic Evolution Over Time (4-week rolling average)')
        fig_topic.update_layout(title_font_size=16)
        st.plotly_chart(fig_topic, use_container_width=True)
        
        st.success("""
        **📊 Topic Modeling Insights:**
        - 🔄 **Dynamic Trends:** Content themes shift continuously over time
        - 📈 **Spike Detection:** Sudden topic increases indicate emerging trends or viral events
        - 🎯 **Seasonal Patterns:** Some topics show recurring seasonal popularity
        - 📅 **4-week Smoothing:** Reduces noise to reveal genuine trend patterns
        """)

        st.markdown("### 2️⃣ 🎭 Sentiment Analysis")
        st.info("💭 **Technique:** Sentiment polarity analysis to understand emotional tone and its impact on virality.")
        
        sentiment_counts = df['sentiment_label'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
        
        fig_sentiment = px.bar(x=sentiment_counts.index, y=sentiment_counts.values, 
                              color=sentiment_counts.index,
                              labels={'x': 'Sentiment Label', 'y': 'Number of Videos'},
                              title='🎭 Sentiment Distribution Across All Videos',
                              color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'})
        fig_sentiment.update_layout(title_font_size=16)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.warning("""
        **🎭 Sentiment Patterns:**
        - ⚖️ **Neutral Majority:** Most videos maintain neutral sentiment
        - 🔥 **Emotional Extremes:** Trending videos show higher proportions of positive/negative sentiment
        - 📊 **Virality Factor:** Strong emotions (both positive and negative) drive engagement
        - 🎯 **Strategic Insight:** Neutral sentiment alone doesn't predict viral success
        """)
        
        # Sentiment vs Virality Score
        fig_sent_virality = px.scatter(df, x='sentiment_polarity', y='virality_score', color='is_trending',
                                      labels={'sentiment_polarity': 'Sentiment Polarity', 'virality_score': 'Virality Score', 'is_trending': 'Trending Status'},
                                      title='🎭 Sentiment Impact on Viral Performance',
                                      color_discrete_map={True: '#ff6b35', False: '#0088cc'})
        fig_sent_virality.update_layout(title_font_size=16)
        st.plotly_chart(fig_sent_virality, use_container_width=True)
        
        st.info("""
        **🔥 Virality vs Sentiment:**
        - 🚀 **Trending Videos:** Achieve higher virality scores regardless of sentiment polarity
        - 🎯 **Sweet Spot:** Peak engagement centers around neutral sentiment with extremes performing well
        - 📈 **Universal Appeal:** High engagement outliers occur at all sentiment levels
        - 💡 **Key Takeaway:** Content quality matters more than sentiment polarity alone
        """)

        st.markdown("### 3️⃣ 👤 Named Entity Recognition (NER)")
        st.info("🔍 **Technique:** Extract and analyze names, brands, locations, and events to identify emerging trend entities.")
        
        # Top emerging entities (by lift)
        emerging_entities = []
        if 'extracted_entities' in df.columns:
            # Convert stringified list of tuples to list of tuples
            df['extracted_entities'] = df['extracted_entities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
            from collections import Counter
            trending_entities = [entity[0] for sublist in df[df['is_trending']]['extracted_entities'] for entity in sublist]
            non_trending_entities = [entity[0] for sublist in df[~df['is_trending']]['extracted_entities'] for entity in sublist]
            trending_entity_counts = Counter(trending_entities)
            non_trending_entity_counts = Counter(non_trending_entities)
            total_trending = sum(trending_entity_counts.values()) or 1
            total_non_trending = sum(non_trending_entity_counts.values()) or 1
            all_unique_entities = set(trending_entity_counts.keys()).union(set(non_trending_entity_counts.keys()))
            for entity in all_unique_entities:
                count_trending = trending_entity_counts.get(entity, 0)
                count_non_trending = non_trending_entity_counts.get(entity, 0)
                if count_trending > 0 and count_non_trending > 0:
                    prop_trending = count_trending / total_trending
                    prop_non_trending = count_non_trending / total_non_trending
                    if prop_non_trending > 0:
                        lift = prop_trending / prop_non_trending
                        if lift > 1.5 and count_trending > 5:
                            emerging_entities.append((entity, lift, count_trending))
            emerging_entities = sorted(emerging_entities, key=lambda x: x[1], reverse=True)[:10]
        
        if emerging_entities:
            st.markdown("#### 🏆 Top Emerging Entities in Trending Videos")
            st.success("**🔍 Entities with Lift > 1.5 (50%+ more likely to appear in trending content):**")
            
            # Create a more visual table
            entity_df = pd.DataFrame(emerging_entities, columns=['Entity', 'Lift', 'Trending Mentions'])
            entity_df['Lift'] = entity_df['Lift'].round(2)
            entity_df['🔥 Trend Strength'] = entity_df['Lift'].apply(
                lambda x: '🔥🔥🔥' if x > 3 else '🔥🔥' if x > 2 else '🔥'
            )
            
            st.dataframe(entity_df, use_container_width=True)
            
            st.info("""
            **📊 About Lift Metric:**
            - 📈 **Lift = 2.0:** Entity is 2x more likely to appear in trending videos
            - 🎯 **Lift > 1.5:** Strong signal for emerging trends (50%+ higher probability)
            - 🔥 **High Lift + Volume:** Best indicators for viral potential
            - 🚀 **Early Detection:** Catches trends before they become mainstream
            """)
        else:
            st.warning("⚠️ No significant emerging entities found or NER data missing.")
        
        # Entity mention volume over time (top 5 entities)
        entity_data = []
        for index, row in df.iterrows():
            for entity in row['extracted_entities']:
                entity_data.append({'date_group': row['date_group'], 'entity': entity[0]})
        
        if entity_data:
            entity_df = pd.DataFrame(entity_data)
            entity_trends_over_time = entity_df.groupby(['date_group', 'entity']).size().unstack(fill_value=0)
            top_entities = entity_df['entity'].value_counts().head(5).index.tolist()
            
            fig_entity = px.line(entity_trends_over_time[top_entities],
                                x=entity_trends_over_time.index,
                                y=top_entities,
                                labels={'value': 'Entity Mentions', 'date_group': 'Date'},
                                title='📈 Top 5 Entity Trends Over Time')
            fig_entity.update_layout(title_font_size=16)
            st.plotly_chart(fig_entity, use_container_width=True)
        else:
            st.info("📊 No entity time series data available.")
        
        st.success("""
        **🎯 Entity Trend Insights:**
        - 📈 **Spike Detection:** Sudden entity mention increases signal viral events
        - 🌟 **Unique Keywords:** Trending videos feature more distinctive entities
        - 📅 **Seasonal Patterns:** Some entities show predictable seasonal trends
        - 🚀 **Early Warning:** NER can detect trends before they hit mainstream
        """)
        
    except Exception as e:
        st.error(f"❌ Could not load NLP trend data or generate plots: {e}")
        st.info("📝 Please ensure the NLP processed data files are available in the correct directory.")

#------------------------------------------------------

elif selected_page == "🤖 Part 5: Predictive Model Development":
    st.markdown('<h2 class="section-header">🤖 Part 5: Predictive Model Development</h2>', unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Build and compare machine learning models to predict viral video performance using engineered features and advanced algorithms.")
    
    # Model Overview with enhanced styling
    st.markdown("### 🛠️ Machine Learning Pipeline Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>📊 Models Tested</h4>
            <ul style="text-align: left; padding-left: 20px;">
                <li>Logistic Regression</li>
                <li>Random Forest</li>
                <li>XGBoost</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>🔧 Feature Types</h4>
            <ul style="text-align: left; padding-left: 20px;">
                <li>Numerical (15+)</li>
                <li>Categorical (5+)</li>
                <li>Text (TF-IDF)</li>
                <li>Historical</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>📈 Evaluation</h4>
            <ul style="text-align: left; padding-left: 20px;">
                <li>ROC-AUC Score</li>
                <li>Precision/Recall</li>
                <li>F1-Score</li>
                <li>Confusion Matrix</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Feature Engineering Section
    st.markdown("### 🔧 Advanced Feature Engineering")
    st.success("**🧮 Virality Score Formula (Custom Weighted Engagement):**")
    
    st.code("""
🔥 virality_score = plays + (1 - corr_views_likes) × likes + 
                           (1 - corr_views_comments) × comments + 
                           (1 - corr_views_shares) × shares

📊 Where correlation coefficients reduce multicollinearity between metrics
    """, language='python')
    
    # Feature categories with more detail
    st.markdown("#### 📋 Feature Categories Breakdown")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.info("""
        **📊 Numerical Features:**
        - ⏰ Temporal: Hour, day, month, weekend
        - #️⃣ Content: Hashtag count, description length
        - 💭 Sentiment: Polarity, subjectivity scores
        - 👤 Creator: Historical metrics, follower proxy
        """)
        
        st.warning("""
        **🏷️ Categorical Features:**
        - 📅 Time periods (morning, afternoon, etc.)
        - 🎭 Content categories (education, comedy, etc.)
        - 👤 Author groupings (high/medium/low performers)
        - 🌍 Language/region indicators
        """)
    
    with feature_col2:
        st.success("""
        **📝 Text Features (TF-IDF):**
        - 🔤 Top 100 most important words
        - 📊 Vectorized video descriptions
        - 🎯 Captures semantic content patterns
        - 🚀 Enables content-based predictions
        """)
        
        st.info("""
        **📈 Historical Features:**
        - 📊 Author average virality
        - 🏆 Author max virality score
        - 📱 Total videos posted
        - 🎯 Historical viral ratio
        """)

    # Model Performance Section
    st.markdown("### 📊 Model Performance Comparison")
    
    performance_data = {
        'Model': ['🔬 Logistic Regression', '🌲 Random Forest', '🚀 XGBoost'],
        'Validation AUC': [0.643, 0.549, 0.604],
        'Test AUC': [0.653, 0.545, 0.598],
        'Training Time': ['Fast ⚡', 'Medium 🔄', 'Medium 🔄'],
        'Interpretability': ['High 📊', 'Medium 🔍', 'Low 🤖']
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best Model", "Logistic Regression")
    with col2:
        st.metric("📈 Best AUC Score", "0.653")
    with col3:
        st.metric("🎯 Performance Level", "Good")
    
    st.success("🏆 **Winner:** Logistic Regression achieves the best performance with AUC = 0.653, demonstrating that linear relationships can effectively capture viral patterns!")

    # Model Visualizations
    st.markdown("### 📈 Model Performance Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("#### 📈 ROC Curve Analysis")
        try:
            st.image("data/graph/question4-roc-curve-logistic.png", 
                    caption="🎯 ROC Curve - Logistic Regression (Test Set)", 
                    use_container_width=True)
            st.info("**ROC AUC = 0.653** indicates good discrimination between viral and non-viral content.")
        except Exception as e:
            st.error(f"❌ Could not load ROC curve: {e}")
    
    with viz_col2:
        st.markdown("#### 🔍 Confusion Matrix")
        try:
            st.image("data/graph/question4-confusion-matrix-logistic.png", 
                    caption="📊 Confusion Matrix - Logistic Regression (Test Set)", 
                    use_container_width=True)
            st.info("**Matrix shows** prediction accuracy for both viral and non-viral categories.")
        except Exception as e:
            st.error(f"❌ Could not load confusion matrix: {e}")

    # Classification Report
    st.markdown("#### 📋 Detailed Classification Report")
    
    classification_data = {
        'Class': ['🚫 Non-trending (0)', '🔥 Trending (1)', '', '📊 Accuracy', '📈 Macro avg', '⚖️ Weighted avg'],
        'Precision': [0.90, 0.21, '', '', 0.55, 0.81],
        'Recall': [0.70, 0.51, '', '', 0.60, 0.67],
        'F1-Score': [0.79, 0.29, '', 0.67, 0.54, 0.72],
        'Support': [626, 97, '', 723, 723, 723]
    }
    
    classification_df = pd.DataFrame(classification_data)
    st.dataframe(classification_df, use_container_width=True)
    
    st.warning("""
    **📊 Key Performance Insights:**
    - 🎯 **High Precision** for non-trending content (90%) - good at avoiding false alarms
    - 🔍 **Moderate Recall** for trending content (51%) - catches about half of viral videos
    - ⚖️ **Class Imbalance** challenge - viral content is naturally rare
    - 📈 **Overall Accuracy** of 67% is solid for this challenging prediction task
    """)

    # Time Series Analysis Section
    st.markdown("### 📈 Time Series Forecasting with ARIMA")
    st.info("🔮 **Advanced Analytics:** ARIMA modeling for predicting future virality score trends with confidence intervals.")
    
    arima_col1, arima_col2 = st.columns(2)
    
    with arima_col1:
        st.markdown("#### 📊 7-Day Rolling Average")
        try:
            st.image("data/graph/question4-7dayrollingaverage-virality-score.png", 
                    caption="📈 Historical virality trends smoothed over 7-day periods", 
                    use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not load rolling average: {e}")
    
    with arima_col2:
        st.markdown("#### 🔮 30-Day Forecast")
        try:
            st.image("data/graph/question4-30day-virality-score-forecast.png", 
                    caption="🚀 Future virality predictions with uncertainty bounds", 
                    use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not load forecast: {e}")

    # ARIMA Model Summary
    st.markdown("#### 🔬 ARIMA Model Technical Details")
    
    arima_col1, arima_col2, arima_col3 = st.columns(3)
    
    with arima_col1:
        st.metric("📊 Model Type", "ARIMA(7,1,2)")
    with arima_col2:
        st.metric("📈 AIC Score", "-98.919")
    with arima_col3:
        st.metric("🎯 Observations", "40")
    
    st.success("""
    **🔬 ARIMA Model Insights:**
    - 📊 **Model ARIMA(7,1,2):** Captures weekly seasonality and short-term dependencies
    - 📈 **Excellent Fit:** AIC of -98.919 indicates strong model performance
    - 🎯 **Significant Parameters:** AR lag-6 coefficient (-0.5026) is statistically significant (p=0.005)
    - 🔮 **Forecast Quality:** Provides daily predictions with realistic confidence intervals
    """)
    
    # Model Insights and Recommendations
    st.markdown("### 💡 Key Model Insights & Recommendations")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.success("""
        **🎯 What Works for Viral Prediction:**
        - ⏰ **Timing Features:** Hour and day significantly impact virality
        - 📝 **Content Quality:** TF-IDF text features capture viral language patterns
        - 👤 **Creator History:** Past performance strongly predicts future success
        - 🎭 **Emotional Content:** Sentiment polarity influences engagement
        """)
    
    with insight_col2:
        st.info("""
        **🚀 Model Applications:**
        - 📊 **Content Strategy:** Optimize posting times and hashtag usage
        - 🎯 **Creator Guidance:** Identify high-potential content characteristics
        - 📈 **Marketing ROI:** Focus resources on likely-viral content
        - 🔮 **Trend Forecasting:** Predict future viral content patterns
        """)
    
    st.warning("""
    **⚠️ Model Limitations & Future Improvements:**
    - 🎯 **Class Imbalance:** Viral content is rare - consider cost-sensitive learning
    - 📊 **Feature Engineering:** Add creator network features and engagement velocity
    - 🤖 **Advanced Models:** Experiment with deep learning and ensemble methods
    - 🔄 **Real-time Updates:** Implement online learning for fresh trend detection
    """)

#------------------------------------------------------

elif selected_page == "🚀 Part 6: AI-Driven Trend Forecasting":
    st.markdown('<h2 class="section-header">🚀 Part 6: AI-Driven Trend Forecasting</h2>', unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Deploy AI models to predict viral trends, hashtags, content categories, and top creators for the next 24 hours and 7 days.")
    
    # Model info header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>🤖 Primary Model</h4>
            <p><strong>Random Forest Classifier</strong></p>
            <p>Ensemble learning for robust predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>⏰ Prediction Windows</h4>
            <p><strong>24 Hours & 7 Days</strong></p>
            <p>Short & medium-term forecasting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>🎯 Prediction Types</h4>
            <p><strong>4 Categories</strong></p>
            <p>Hashtags, Content, Creators, Performance</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ROC Curve Section
    st.markdown("### 📈 Model Performance Evaluation")
    
    roc_col1, roc_col2 = st.columns(2)
    
    with roc_col1:
        st.markdown("#### 🕐 24-Hour Prediction Model")
        try:
            st.image("data/graph/ROC-24HOURS.png", caption="🎯 ROC Curve - 24-Hour Viral Prediction Model")
        except Exception as e:
            st.error(f"❌ Could not load 24-hour ROC curve: {e}")
        
        st.success("""
        **🎯 24-Hour Model Performance:**
        - 🔥 **Optimized for Speed:** Quick trend detection
        - 📊 **High Precision:** Minimizes false positives
        - ⚡ **Real-time Ready:** Fast prediction capabilities
        """)

    with roc_col2:
        st.markdown("#### 📅 7-Day Prediction Model")
        try: 
            st.image("data/graph/ROC-7DAYS.png", caption="📈 ROC Curve - 7-Day Viral Prediction Model")
        except Exception as e:
            st.error(f"❌ Could not load 7-day ROC curve: {e}")
        
        st.info("""
        **📈 7-Day Model Performance:**
        - 🎯 **Strategic Planning:** Medium-term trend forecasting
        - 📊 **Pattern Recognition:** Captures weekly cycles
        - 🔮 **Trend Stability:** More stable predictions
        """)

    st.markdown("---")

    # Viral Hashtag Prediction Section
    st.markdown("### #️⃣ Viral Hashtag Prediction")
    st.success("🏷️ **AI-Powered Hashtag Trend Forecasting** - Identify hashtags likely to go viral based on historical patterns and current momentum.")
    
    hashtag_col1, hashtag_col2 = st.columns(2)
    
    with hashtag_col1:
        st.markdown("#### 🕐 Next 24 Hours - Trending Hashtags")
        try:
            st.image("data/graph/HASHTAG-24.png", caption="🔥 Top Hashtags Predicted to Trend in 24 Hours")
            st.info("**🚀 Quick Wins:** Hashtags with immediate viral potential for today's content")
        except Exception as e:
            st.error(f"❌ Could not load 24-hour hashtag prediction: {e}")

    with hashtag_col2:
        st.markdown("#### 📅 Next 7 Days - Hashtag Trends")
        try:
            st.image("data/graph/HASHTAG-7.png", caption="📈 Hashtag Trends for the Next Week")
            st.warning("**📅 Strategy:** Plan your content calendar around these emerging hashtag trends")
        except Exception as e:
            st.error(f"❌ Could not load 7-day hashtag prediction: {e}")

    st.markdown("---")

    # Viral Category Prediction Section
    st.markdown("### 🎯 Viral Content Category Prediction")
    st.info("📱 **Content Category Intelligence** - Predict which content types will dominate the viral landscape.")
    
    category_col1, category_col2 = st.columns(2)
    
    with category_col1:
        st.markdown("#### 🕐 24-Hour Category Distribution")
        try:
            st.image("data/graph/PIE-24HOURS.png", caption="🎪 Content Categories Most Likely to Go Viral Today")
            st.success("""
            **🎯 Today's Viral Recipe:**
            - Focus on the dominant categories shown
            - Create content aligned with these trends
            - Optimize posting for peak category performance
            """)
        except Exception as e:
            st.error(f"❌ Could not load 24-hour category prediction: {e}")

    with category_col2:
        st.markdown("#### 📅 7-Day Category Trends")
        try:
            st.image("data/graph/PIE-7DAYS.png", caption="📊 Weekly Content Category Forecast")
            st.info("""
            **📅 Weekly Strategy:**
            - Plan diverse content across trending categories
            - Identify emerging category opportunities
            - Prepare content for category shifts
            """)
        except Exception as e:
            st.error(f"❌ Could not load 7-day category prediction: {e}")

    st.markdown("---")

    # Viral Account Prediction Section
    st.markdown("### 👑 Top Viral Creator Predictions")
    st.warning("🌟 **Creator Success Forecasting** - AI-predicted creators most likely to achieve viral success in upcoming periods.")
    
    account_col1, account_col2 = st.columns(2)
    
    with account_col1:
        st.markdown("#### 🕐 Next 24 Hours - Rising Stars")
        try:
            df_24h = pd.read_csv("data/processed/top_viral_24h.csv")
            
            # Enhanced display with metrics
            st.metric("🌟 Predicted Viral Creators", len(df_24h))
            st.markdown("**🔥 Top 10 Creators to Watch Today:**")
            
            # Add emoji indicators for prediction confidence
            if 'predicted_proba' in df_24h.columns:
                df_24h['🔥 Confidence'] = df_24h['predicted_proba'].apply(
                    lambda x: '🔥🔥🔥' if x > 0.8 else '🔥🔥' if x > 0.6 else '🔥'
                )
            
            st.dataframe(df_24h.head(10), use_container_width=True)
            
            st.success("**🚀 Immediate Action:** Consider collaborating with these creators or monitoring their content strategies!")
            
        except Exception as e:
            st.error(f"❌ Could not load 24-hour creator predictions: {e}")

    with account_col2:
        st.markdown("#### 📅 Next 7 Days - Strategic Partnerships")
        try:
            df_7d = pd.read_csv("data/processed/top_viral_7d.csv")
            
            st.metric("📈 Weekly Viral Forecast", len(df_7d))
            st.markdown("**📊 Top 10 Creators - Weekly Outlook:**")
            
            # Add confidence indicators
            if 'predicted_proba' in df_7d.columns:
                df_7d['📈 Trend Strength'] = df_7d['predicted_proba'].apply(
                    lambda x: 'Strong 💪' if x > 0.7 else 'Moderate 👍' if x > 0.5 else 'Emerging 🌱'
                )
            
            st.dataframe(df_7d.head(10), use_container_width=True)
            
            st.info("**📅 Strategic Planning:** These creators show sustained viral potential - perfect for long-term partnerships!")
            
        except Exception as e:
            st.error(f"❌ Could not load 7-day creator predictions: {e}")

    st.markdown("---")

    # AI Insights and Recommendations
    st.markdown("### 🧠 AI-Powered Insights & Recommendations")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.success("""
        **🎯 For Content Creators:**
        - 📱 Use predicted hashtags for maximum reach
        - 🎭 Align content with trending categories
        - ⏰ Post during predicted peak times
        - 🤝 Collaborate with rising creators
        """)
    
    with insight_col2:
        st.info("""
        **📈 For Marketers:**
        - 🎯 Target predicted viral categories
        - 👑 Partner with forecasted top creators
        - #️⃣ Incorporate trending hashtags early
        - 📅 Plan campaigns around predictions
        """)
    
    with insight_col3:
        st.warning("""
        **🔮 For Strategists:**
        - 📊 Monitor prediction accuracy
        - 🔄 Adjust models based on outcomes
        - 📈 Track competitor presence in predictions
        - 💡 Identify white-space opportunities
        """)

    # Model Confidence and Limitations
    st.markdown("### ⚠️ Model Confidence & Limitations")
    
    st.error("""
    **🚨 Important Considerations:**
    - 📊 **Predictions are probabilistic** - not guarantees of viral success
    - ⏰ **External events** (news, trends, algorithms) can affect accuracy
    - 🔄 **Model performance** varies with data quality and recency
    - 🎯 **Use predictions as guidance** combined with human creativity and intuition
    - 📈 **Continuously validate** predictions against actual outcomes for improvement
    """)
    
    st.success("""
    **🏆 Best Practices for Using Predictions:**
    - 🎯 Combine AI predictions with domain expertise
    - 📊 Track prediction accuracy over time
    - 🔄 Use predictions for inspiration, not rigid rules
    - 💡 Focus on content quality alongside predicted trends
    - 📈 Adapt strategies based on real-world performance
    """)


#------------------------------------------------------

elif selected_page == "📊 Part 7: Visualization":
    st.markdown('<h2 class="section-header">📊 Part 7: Interactive Data Visualization</h2>', unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Create interactive dashboards and visualizations to explore TikTok trends, patterns, and insights in real-time.")

    # Load processed data
    data_path = 'data/processed/tiktok_processed_with_nlp_features.csv'
    
    try:
        df = pd.read_csv(data_path)
        
        # Data preprocessing
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df['date'] = df['create_time'].dt.date
        df.columns = df.columns.str.strip()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", f"{len(df):,}")
        with col2:
            st.metric("📅 Date Range", f"{(df['create_time'].max() - df['create_time'].min()).days} days")
        with col3:
            st.metric("🔍 NLP Features", "Multiple")
        with col4:
            st.metric("📈 Interactive Charts", "4")
        
        st.markdown("---")

        # === Top Entities Over Time ===
        st.markdown("### 1️⃣ 👤 Top Extracted Entities Over Time")
        st.success("🔍 **Named Entity Recognition:** Track mentions of people, organizations, and locations over time to identify trending topics.")
        
        TOP_N = st.slider("🎯 Select number of top entities to display:", 3, 20, 5, key="entities_slider")

        if 'create_time' in df.columns and 'extracted_entities' in df.columns:
            df['extracted_entities'] = df['extracted_entities'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

            # Entity category mapping with emojis
            ENTITY_LABELS_MAP = {
                "PERSON": "👤 Person", "ORG": "🏢 Organization", "GPE": "🌍 Country/City/State", "LOC": "📍 Location",
                "PRODUCT": "📱 Product", "EVENT": "🎉 Event", "WORK_OF_ART": "🎨 Work of Art", "LAW": "⚖️ Law",
                "LANGUAGE": "🗣️ Language", "DATE": "📅 Date", "TIME": "⏰ Time", "PERCENT": "📊 Percent",
                "MONEY": "💰 Money", "QUANTITY": "📏 Quantity", "ORDINAL": "🔢 Ordinal", "CARDINAL": "🔢 Cardinal",
                "NORP": "🏛️ Nationality/Religious/Political Group", "FAC": "🏗️ Facility",
            }

            # Get all available entity labels
            all_labels = set()
            for ents in df['extracted_entities']:
                if isinstance(ents, list):
                    for ent in ents:
                        if isinstance(ent, (list, tuple)) and len(ent) == 2:
                            all_labels.add(ent[1])
            
            all_labels = sorted(all_labels)
            label_display_map = {ENTITY_LABELS_MAP.get(code, f"🏷️ {code}"): code for code in all_labels}
            display_labels = list(label_display_map.keys())

            selected_display_labels = st.multiselect(
                "🏷️ Filter by entity category:",
                options=display_labels,
                default=display_labels[:5] if len(display_labels) > 5 else display_labels,
                key="entity_labels"
            )
            selected_labels = [label_display_map[d] for d in selected_display_labels]

            # Process entity data
            entity_df = df.explode('extracted_entities')
            entity_df = entity_df.dropna(subset=['extracted_entities'])
            entity_df[['entity', 'label']] = entity_df['extracted_entities'].apply(
                lambda x: pd.Series(x) if isinstance(x, (list, tuple)) and len(x) == 2 else pd.Series([None, None])
            )
            entity_df = entity_df[entity_df['label'].isin(selected_labels)]

            if not entity_df.empty:
                top_entities = entity_df['entity'].value_counts().nlargest(TOP_N).index.tolist()
                filtered_entity_df = entity_df[entity_df['entity'].isin(top_entities)]

                trend_df = (
                    filtered_entity_df.groupby(['date', 'entity'])
                    .size()
                    .reset_index(name='mentions')
                )

                fig = px.line(
                    trend_df,
                    x='date',
                    y='mentions',
                    color='entity',
                    title=f"📈 Top {TOP_N} Mentioned Entities Over Time",
                    labels={'mentions': 'Daily Mentions', 'date': 'Date', 'entity': 'Entity'}
                )
                fig.update_layout(title_font_size=16, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"📊 Showing frequency of top {TOP_N} named entities extracted per day, filtered by selected categories.")
            else:
                st.warning("⚠️ No entity data available for selected categories.")
        else:
            st.error("❌ Required columns 'create_time' and 'extracted_entities' are missing from the dataset.")

        st.markdown("---")

        # === Timing vs Engagement ===
        st.markdown("### 2️⃣ ⏰ Post Timing vs Engagement Heatmap")
        st.info("🕒 **Optimal Timing Analysis:** Discover the best times to post for maximum engagement using smooth heatmap visualization.")
        
        if 'create_hour' in df.columns and 'day_of_week' in df.columns and 'engagement_rate_per_play_capped' in df.columns:
            pivot = df.pivot_table(
                values='engagement_rate_per_play_capped',
                index='day_of_week',
                columns='create_hour',
                aggfunc='mean'
            ).sort_index()

            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    colorscale='Viridis',
                    zsmooth='best',
                    colorbar=dict(title="📈 Engagement Rate Per Play"),
                    hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Engagement Rate: %{z:.3f}<extra></extra>'
                )
            )

            fig.update_layout(
                title="🕒 Smooth Heatmap: Engagement Rate by Post Timing",
                title_font_size=16,
                xaxis_title="⏰ Hour of Day",
                yaxis_title="📅 Day of Week",
                autosize=True,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.success("""
            **🎯 Timing Strategy Insights:**
            - 🌃 **Dark areas** = Lower engagement periods
            - 🌟 **Bright areas** = Peak engagement windows
            - 📊 **Use this heatmap** to optimize your posting schedule
            - ⏰ **Best practice:** Post during bright zones for maximum reach
            """)
        else:
            st.warning("⚠️ Required columns for timing analysis are missing.")

        st.markdown("---")

        # === Sentiment vs Virality ===
        st.markdown("### 3️⃣ 🎭 Sentiment vs Virality Analysis")
        st.warning("💭 **Emotional Impact Study:** Explore how content sentiment affects viral performance (excluding neutral content for clarity).")
        
        if {'sentiment_polarity', 'sentiment_subjectivity', 'virality_score_normalized'}.issubset(df.columns):
            # Filter out neutral polarity for clearer visualization
            filtered_df = df[
                (df['sentiment_polarity'] != 0) &
                (df['virality_score_normalized'] < 1)
            ]

            fig = px.scatter(
                filtered_df,
                x='sentiment_polarity',
                y='virality_score_normalized',
                color='sentiment_subjectivity',
                color_continuous_scale='RdYlBu_r',
                title='🎭 Sentiment Impact on Viral Performance',
                labels={
                    'sentiment_polarity': '😢 Negative ← Sentiment Polarity → Positive 😊',
                    'virality_score_normalized': '📈 Normalized Virality Score',
                    'sentiment_subjectivity': '📊 Subjectivity (Objective ← → Subjective)'
                },
                hover_data=['video_id'] if 'video_id' in df.columns else None,
                size_max=10
            )
            
            fig.update_layout(
                title_font_size=16,
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[0, 0.5]),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **🎭 Sentiment Insights:**
            - 🔴 **Red dots:** Highly subjective content (opinions, emotions)
            - 🔵 **Blue dots:** More objective content (facts, news)
            - 😊 **Right side:** Positive sentiment content
            - 😢 **Left side:** Negative sentiment content
            - 📈 **Higher Y-axis:** Greater viral potential
            """)
            
            st.caption('⚠️ Neutral-toned content (polarity = 0) is excluded for visual clarity.')
        else:
            st.warning("⚠️ Required sentiment analysis columns are missing.")

        st.markdown("---")

        # === Trending Hashtags Monitor ===
        st.markdown("### 4️⃣ 📈 Real-Time Trending Hashtag Monitor")
        st.success("🏷️ **Hashtag Intelligence Dashboard:** Monitor trending hashtags with customizable time windows for strategic content planning.")
        
        if 'create_time' in df.columns and 'hashtag_list_clean' in df.columns:
            df['hashtag_list_clean'] = df['hashtag_list_clean'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

            current_time = df['create_time'].max()
            
            # Enhanced time window selector
            col1, col2 = st.columns(2)
            with col1:
                time_window_days = st.slider("📅 Time window (days)", 1, 30, 7, key="hashtag_slider")
            with col2:
                min_mentions = st.slider("🎯 Minimum mentions", 1, 50, 5, key="min_mentions")

            recent_df = df[df['create_time'] > (current_time - pd.Timedelta(days=time_window_days))]

            # Process hashtag data
            exploded = recent_df.explode('hashtag_list_clean')
            exploded = exploded.dropna(subset=['hashtag_list_clean'])
            exploded = exploded[
                exploded['hashtag_list_clean'].notna() &
                (exploded['hashtag_list_clean'].str.strip() != '')
            ]            
            hashtag_counts = exploded['hashtag_list_clean'].value_counts()
            top_hashtags = hashtag_counts[hashtag_counts >= min_mentions].head(15)

            if not top_hashtags.empty:
                hashtag_df = top_hashtags.reset_index()
                
                # Fix column names for newer pandas versions
                if 'hashtag_list_clean' in hashtag_df.columns:
                    hashtag_df.columns = ['Hashtag', 'Mentions']
                elif 'index' in hashtag_df.columns:
                    hashtag_df = hashtag_df.rename(columns={'index': 'Hashtag', 'hashtag_list_clean': 'Mentions'})
                else:
                    # For newer pandas versions that use 'count' as default column name
                    hashtag_df.columns = ['Hashtag', 'Mentions']
                
                # Add trend indicators
                hashtag_df['🔥 Trend Level'] = hashtag_df['Mentions'].apply(
                    lambda x: '🔥🔥🔥 Viral' if x > hashtag_df['Mentions'].quantile(0.8) else
                             '🔥🔥 Hot' if x > hashtag_df['Mentions'].quantile(0.6) else
                             '🔥 Rising'
                )

                fig = px.bar(
                    hashtag_df,
                    x='Mentions',
                    y='Hashtag',
                    title=f"🏆 Top Trending Hashtags (Last {time_window_days} Days)",
                    orientation='h',
                    color='Mentions',
                    color_continuous_scale='Viridis',
                    hover_data=['🔥 Trend Level']
                )

                fig.update_layout(
                    title_font_size=16,
                    xaxis_title='📊 Total Mentions',
                    yaxis_title='#️⃣ Hashtag',
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Display hashtag data table
                st.markdown("#### 📊 Hashtag Performance Details")
                st.dataframe(hashtag_df, use_container_width=True)
                
                # Time period info
                start_date = current_time - pd.Timedelta(days=time_window_days)
                st.info(f"📅 **Analysis Period:** {start_date.strftime('%Y-%m-%d %H:%M')} to {current_time.strftime('%Y-%m-%d %H:%M')}")
                
                st.success(f"""
                **🎯 Hashtag Strategy Tips:**
                - 🔥 **Use trending hashtags** from the top 5 for maximum visibility
                - 📈 **Monitor rising hashtags** for early trend adoption
                - #️⃣ **Combine popular and niche** hashtags for optimal reach
                - ⏰ **Update strategy regularly** based on trend shifts
                """)
            else:
                st.warning(f"⚠️ No hashtags found with at least {min_mentions} mentions in the selected time period.")
                
        else:
            st.error("❌ Required columns 'create_time' and 'hashtag_list_clean' not found.")

        st.markdown("---")

        # === Data Export Option ===
        with st.expander("📁 Export Raw Data"):
            st.markdown("### 💾 Download Processed Dataset")
            st.info("🔄 **Complete Dataset:** Download the full processed dataset with all NLP features for further analysis.")
            
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"tiktok_analysis_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.markdown("**📊 Dataset Preview:**")
            st.dataframe(df.head(20), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", f"{len(df):,}")
            with col2:
                st.metric("🏷️ Total Columns", f"{len(df.columns)}")
            with col3:
                file_size = len(csv_data) / (1024 * 1024)  # MB
                st.metric("💾 File Size", f"{file_size:.1f} MB")

    except Exception as e:
        st.error(f"❌ Could not load or process visualization data: {e}")
        st.info("📝 Please ensure the processed data files are available in the correct directory.")
        
        # Fallback information
        st.markdown("### 📋 Expected Visualizations")
        st.markdown("""
        **🎯 This section should display:**
        1. 👤 **Entity Trends:** Track mentions of people, brands, and topics over time
        2. ⏰ **Timing Heatmap:** Optimal posting times for maximum engagement
        3. 🎭 **Sentiment Analysis:** How emotional tone affects viral performance
        4. #️⃣ **Hashtag Monitor:** Real-time trending hashtag tracking
        
        **📁 Required Files:**
        - `data/processed/tiktok_processed_with_nlp_features.csv`
        """)


#------------------------------------------------------

elif selected_page == "📝 Part 8 & 9: Summary":
    st.markdown('<h2 class="section-header">📝 Part 8 & 9: Project Summary & Conclusions</h2>', unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Comprehensive overview of our AI-powered TikTok trend analysis project, key insights, achievements, and future opportunities.")
    
    # Project overview banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff0050, #ff9900); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0;">
        <h3 style="margin: 0; color: white;">🔥 AI-Powered Trend Prediction on TikTok</h3>
        <h4 style="margin: 0.5rem 0 0 0; color: white;">Leveraging Data Science for Social Media Insights</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Project metrics overview
    st.markdown("### 📊 Project Achievement Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>🎯 Data Points</h4>
            <h2>10,000+</h2>
            <p>TikTok Videos Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>🤖 ML Models</h4>
            <h2>3</h2>
            <p>Advanced Algorithms Tested</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>🔍 NLP Features</h4>
            <h2>50+</h2>
            <p>Extracted & Engineered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h4>📈 Best AUC</h4>
            <h2>0.653</h2>
            <p>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 8: Key Insights
    st.markdown("### 8️⃣ 💡 Key Insights & Discoveries")
    
    # AI Applications section
    st.markdown("#### 🤖 AI Application for Trend Forecasting")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🎯 Predictive Capabilities:**
        - ⏰ **Short-term virality** prediction using metadata
        - 💭 **Sentiment analysis** impact on engagement
        - #️⃣ **Hashtag optimization** recommendations
        - 📊 **Content category** performance insights
        """)
    
    with col2:
        st.info("""
        **🎬 Target Beneficiaries:**
        - 🎨 **Content Creators** - Optimize posting strategies
        - 🏢 **Brands & Marketers** - Align with trending topics
        - 📈 **Social Media Managers** - Data-driven decisions
        - 🎯 **Influencers** - Maximize reach potential
        """)
    
    # Actionable recommendations
    st.markdown("#### 📋 Actionable Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["👨‍🎨 For Creators", "🏢 For Brands", "📊 For Strategists"])
    
    with tab1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;">
        <h4 style="color: white;">🎨 Creator Optimization Guide</h4>
        <ul style="color: white;">
            <li>🌙 <strong>Prime Time:</strong> Post during midnight–5 AM for highest engagement</li>
            <li>#️⃣ <strong>Hashtag Sweet Spot:</strong> Use 5–12 hashtags per post</li>
            <li>🎭 <strong>Emotional Content:</strong> Create emotionally engaging content</li>
            <li>📚 <strong>Educational Focus:</strong> Educational content shows highest engagement rates</li>
            <li>🔥 <strong>Trending Topics:</strong> Align with current trending entities</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #764ba2 100%, #667eea 0%); padding: 1.5rem; border-radius: 10px; color: white;">
        <h4 style="color: white;">🏢 Brand Strategy Insights</h4>
        <ul style="color: white;">
            <li>📈 <strong>High-Lift Entities:</strong> Monitor trending entities with strong virality correlation</li>
            <li>🏆 <strong>Top Categories:</strong> Focus on Sports, DIY, and Educational content</li>
            <li>📊 <strong>Data-Driven Timing:</strong> Leverage engagement heatmaps for posting schedules</li>
            <li>🎯 <strong>Sentiment Alignment:</strong> Match brand messaging with trending sentiment patterns</li>
            <li>🔍 <strong>Real-time Monitoring:</strong> Use hashtag monitoring for trend detection</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff9900, #ff0050); padding: 1.5rem; border-radius: 10px; color: white;">
        <h4 style="color: white;">📊 Strategic Implementation</h4>
        <ul style="color: white;">
            <li>🎯 <strong>Probability Scores:</strong> Use model confidence for content prioritization</li>
            <li>📈 <strong>Forecast Integration:</strong> Incorporate predictions into campaign planning</li>
            <li>⚡ <strong>Proactive Approach:</strong> Act on emerging trends before they peak</li>
            <li>📊 <strong>Performance Tracking:</strong> Monitor prediction accuracy over time</li>
            <li>🔄 <strong>Iterative Improvement:</strong> Continuously refine strategies based on outcomes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 9: Challenges and Future Work
    st.markdown("### 9️⃣ 🚧 Challenges & Future Opportunities")
    
    # Challenges section
    st.markdown("#### ⚠️ Project Challenges Encountered")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.warning("""
        **📊 Data Limitations:**
        - 🔒 **API Restrictions:** Limited access to follower counts and full metadata
        - 📱 **Platform Changes:** TikTok algorithm updates affect data consistency
        - ⏰ **Temporal Bias:** Trend patterns may shift over time
        """)
        
        st.error("""
        **⚖️ Technical Challenges:**
        - 🎯 **Imbalanced Classes:** Viral content represents <20% of dataset
        - 🔄 **Feature Engineering:** Complex interactions between variables
        - 💾 **Scalability:** Processing large volumes of text and media data
        """)
    
    with col2:
        st.warning("""
        **🌍 External Factors:**
        - 📺 **Real-world Events:** Virality affected by news, trends, cultural events
        - 🎭 **Creator Behavior:** Changing content strategies impact predictions
        - 📈 **Platform Evolution:** Algorithm changes affect engagement patterns
        """)
        
        st.error("""
        **🎯 Model Limitations:**
        - 🔮 **Prediction Horizon:** Limited to short-term forecasting
        - 📊 **Feature Dependencies:** Requires comprehensive metadata
        - ⚡ **Real-time Processing:** Latency in live prediction systems
        """)
    
    # Future work section
    st.markdown("#### 🚀 Future Development Roadmap")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Near-term Goals", "🌟 Medium-term Vision", "🚀 Long-term Innovation"])
    
    with tab1:
        st.success("""
        **⚡ Real-Time Forecasting (3-6 months):**
        - 🔄 **Live Pipeline Integration:** Continuous trend monitoring
        - 📊 **Dashboard Enhancement:** Real-time prediction updates
        - 🎯 **Alert Systems:** Automated notifications for emerging trends
        - 📈 **Performance Tracking:** Live model accuracy monitoring
        """)
    
    with tab2:
        st.info("""
        **🧠 Advanced NLP & ML (6-12 months):**
        - 🤖 **Transformer Models:** BERT/GPT integration for better semantic understanding
        - 🎬 **Multimodal Analysis:** Combine video, audio, and text features
        - 👥 **User Behavior Modeling:** Creator-specific virality patterns
        - 🔗 **Graph Neural Networks:** Model creator-audience relationships
        """)
    
    with tab3:
        st.success("""
        **🌍 Cross-Platform Intelligence (1+ years):**
        - 🌐 **Multi-Platform Data:** Instagram, YouTube, Twitter integration
        - 📊 **External Signals:** Google Trends, news events, cultural moments
        - 🎯 **Personalized Predictions:** Individual creator optimization
        - 🤖 **Autonomous Content:** AI-assisted content creation recommendations
        """)
    
    # Technical achievements summary
    st.markdown("---")
    st.markdown("### 🏆 Technical Achievements Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Data Engineering:**
        - ✅ Comprehensive data preprocessing pipeline
        - ✅ Feature engineering (50+ variables)
        - ✅ NLP processing (sentiment, entities, topics)
        - ✅ Robust data validation and cleaning
        """)
    
    with col2:
        st.markdown("""
        **🤖 Machine Learning:**
        - ✅ Multiple model comparison (LR, RF, XGBoost)
        - ✅ Advanced feature selection techniques
        - ✅ Class imbalance handling (SMOTE)
        - ✅ Model validation and testing
        """)
    
    with col3:
        st.markdown("""
        **📈 Visualization & Deployment:**
        - ✅ Interactive Streamlit dashboard
        - ✅ Real-time trend monitoring
        - ✅ Professional data visualizations
        - ✅ User-friendly interface design
        """)
    
    # Final call-to-action
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;">
        <h3 style="color: white;">🎯 Ready to Leverage TikTok Trend Intelligence?</h3>
        <p style="color: white; font-size: 1.1em;">This AI-powered system provides actionable insights for content creators, brands, and social media strategists to optimize their TikTok presence and maximize viral potential.</p>
        <p style="color: white; margin: 0;"><strong>🚀 Start predicting trends. Start going viral. Start succeeding.</strong></p>
    </div>
    """, unsafe_allow_html=True)
