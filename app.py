import streamlit as st
import pandas as pd
import numpy as np
import ast
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    "Part 1: Data Collection",
    "Part 2: Data Preprocessing",
    "Part 3: Exploratory Data Analysis (EDA)",
    "Part 4: Trend Identification Using NLP",
    "Part 5: Predictive Model Development",
    "Part 6: AI-Driven Trend Forecasting",
    "Part 7: Visualization",
    "Part 8 & 9: Summary"
]
selected_page = st.sidebar.radio("Go to", pages)

# Main content
st.title("AI-Powered Trend Prediction on TikTok")

if selected_page == "Part 1: Data Collection":
    st.header("Part 1: Data Collection")
    st.write("Dataset Source: FastAPI TikTok Scraper")
    st.write("Sample of the collected TikTok data:")
    

    try:
        df = pd.read_csv("data/merged/merged_data_deduplicated.csv")
        st.dataframe(df.head(30))  # Show first 5 rows
        st.write("**Feature Descriptions:**")
        feature_table = """
| Feature      | Description                                      |
|--------------|--------------------------------------------------|
| video_id     | Unique identifier for the TikTok video           |
| author       | Username of the video creator                    |
| description  | Caption or text description of the video         |
| likes        | Number of likes the video has received           |
| comments     | Number of comments on the video                  |
| shares       | Number of times the video has been shared        |
| plays        | Number of times the video has been played        |
| hashtags     | List of hashtags used in the video description   |
| music        | Name or ID of the music used in the video        |
| create_time  | Timestamp when the video was created             |
| video_url    | Direct URL link to the video                     |
| fetch_time   | Timestamp when the data was collected            |
| views        | Number of views the video has received           |
| posted_time  | Time when the video was posted                   |
"""
        st.markdown(feature_table)
    except Exception as e:
        st.error(f"Could not load data: {e}")

#------------------------------------------------------

elif selected_page == "Part 2: Data Preprocessing":
    st.header("Part 2: Data Cleaning and Preprocessing")

    st.header("1. Column Merging & Null Value Handling")

    try:
        df = pd.read_csv("data/merged/merged_data_deduplicated.csv")
        null_table = pd.DataFrame({
            "Feature": df.columns,
            "Null Value": df.isnull().sum().values
        })
        st.dataframe(null_table)
    except Exception as e:
        st.error(f"Could not load data: {e}")

    st.subheader("**1.1 Column Merging**")

    st.markdown("""
    - posted_time & create_time --> create_time
    - views & plays --> plays
    """)

    df['create_time'] = df['create_time'].fillna(df['posted_time'])
    df['plays'] = df['plays'].fillna(df['views'])
    st.write("\nNull values in create_time:", df['create_time'].isnull().sum())
    st.write("Null values in plays:", df['plays'].isnull().sum())

    st.subheader("1.2 Null Value Handling")
    st.markdown("""
- Description null value --> No Description
- Hashtags null value --> empty list [  ]
""")


    try:
            df = pd.read_csv("data/processed/null_value_handling.csv")
            st.dataframe(df.head(10))
    except Exception as e:
            st.error(f"Could not load data: {e}")

    st.header("2. Description cleaning & Hashtag processing")
    st.subheader("2.1 Description cleaning")
    st.write("|  Remove emojis, links, and special characters from description")

    try:
            df = pd.read_csv("data/processed/clean_desc_sample.csv")
            st.dataframe(df.head(5))
    except Exception as e:
            st.error(f"Could not load data: {e}")

    st.subheader("2.2 Hashtag processing")
    st.write("|  Change hashtags to list")

    try:
            df = pd.read_csv("data/processed/clean_hashtags_sample.csv")
            st.dataframe(df.head(5))
    except Exception as e:
            st.error(f"Could not load data: {e}")



    st.header("3. Engagement Metrics Calculation")
    st.write("| Create engagement metrics to quantify video performance")
    st.markdown("""
    - Total engagement = likes + comments + shares
    - Engagement rate per play = total engagement / plays
    """)


    try:
            df = pd.read_csv("data/processed/engagement_metric_sample.csv")
            st.dataframe(df.head(5))
    except Exception as e:
            st.error(f"Could not load data: {e}")


    st.header("4. NLP Hashtags Extraction")
    st.write(" Top 20 hashtags from the dataset")

    try:
        st.image("data/graph/hashtag_wordcloud.png")
    except Exception as e:
        st.error(f"Could not load image: {e}")


    st.header("5. Cleaned Data Sample")
    try:
        df = pd.read_csv("data/processed/tiktok_processed_sample.csv")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Could not load data: {e}")

#------------------------------------------------------

elif selected_page == "Part 3: Exploratory Data Analysis (EDA)":
    st.header("Part 3: Exploratory Data Analysis (EDA)")
    st.write("Explore the data with interactive visualizations and statistics.")
    try:
        import plotly.express as px
        df = pd.read_csv("data/processed/tiktok_categorized.csv")
        st.subheader("1. Engagement Rate per Play by Hour of Day")
        fig1 = px.box(df, x='create_hour', y='engagement_rate_per_play',
                     title="Engagement Rate per Play by Hour of Day",
                     labels={"create_hour": "Hour of Day", "engagement_rate_per_play": "Engagement Rate"})
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("""
**Observations:**
- Median engagement rate is stable across hours (~0.10 to ~0.15).
- Slightly higher engagement between midnight to 5 AM.
- Outliers present at all hours (viral potential anytime).
- Reduced engagement around 10 AM – 12 PM.
""")

        st.subheader("2. Engagement Rate per Play by Day of Week")
        fig2 = px.box(df, x='day_of_week', y='engagement_rate_per_play',
                     title="Engagement Rate per Play by Day of Week",
                     labels={"day_of_week": "Day of Week (0=Monday)", "engagement_rate_per_play": "Engagement Rate"})
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
**Observations:**
- Sunday shows the highest median engagement.
- Monday is a close second.
- Midweek days (Tue–Fri) have lower medians.
- Outliers occur on all days (viral content possible any day).
""")

        st.subheader("3. Sentiment Polarity vs. Engagement Rate")
        fig3 = px.scatter(df, x='sentiment_polarity', y='engagement_rate_per_play',
                         color='sentiment_subjectivity',
                         color_continuous_scale='RdBu',
                         title="Sentiment Polarity vs. Engagement Rate",
                         labels={"sentiment_polarity": "Sentiment Polarity", "engagement_rate_per_play": "Engagement Rate per Play", "sentiment_subjectivity": "Subjectivity"})
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
**Key Interpretations:**
- Engagement occurs across all sentiment values.
- Neutral sentiment dominates volume, but less likely to reach high engagement extremes.
- Highly subjective content (redder dots) is widely spread and can perform well.
- High engagement occurs at all sentiment levels.
""")

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



elif selected_page == "Part 4: Trend Identification Using NLP":
    st.header("Part 4: Trend Identification Using NLP")
    st.write("Use NLP techniques to identify trends in TikTok video descriptions and hashtags.")
    try:
        import plotly.express as px
        import pandas as pd
        import ast
        df = pd.read_csv("data/processed/tiktok_processed_with_nlp_features.csv")
        st.subheader("1. Topic Modeling (LDA) Results")
        st.write("Each video description is assigned a dominant topic using Latent Dirichlet Allocation (LDA). The distribution of topics over time helps identify emerging trends.")
        # Topic evolution over time (weekly)
        df['create_time'] = pd.to_datetime(df['create_time'])
        df['date_group'] = df['create_time'].dt.to_period('W').dt.start_time
        topic_evolution = df.groupby('date_group')['dominant_topic'].value_counts(normalize=True).unstack(fill_value=0)
        topic_evolution = topic_evolution.rolling(window=4).mean().dropna()
        fig_topic = px.line(topic_evolution, x=topic_evolution.index, y=topic_evolution.columns,
                           labels={'value': 'Topic Proportion', 'date_group': 'Date'},
                           title='Topic Evolution Over Time (4-week rolling average)')
        st.plotly_chart(fig_topic, use_container_width=True)
        st.markdown("""
**Insights:**
- Topic modeling reveals how content themes shift over time.
- Spikes in certain topics may indicate emerging trends or viral events.
""")

        st.subheader("2. Sentiment Analysis")
        st.write("Sentiment polarity is calculated for each video description. Sentiment labels (Positive, Neutral, Negative) are assigned based on polarity.")
        sentiment_counts = df['sentiment_label'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
        fig_sentiment = px.bar(x=sentiment_counts.index, y=sentiment_counts.values, color=sentiment_counts.index,
                              labels={'x': 'Sentiment Label', 'y': 'Number of Videos'},
                              title='Sentiment Label Distribution')
        st.plotly_chart(fig_sentiment, use_container_width=True)
        st.markdown("""
**Observations:**
- Most videos are Neutral, but trending videos have a higher proportion of Positive and Negative sentiment.
- Neutral sentiment is not a strong predictor of virality on its own.
""")
        # Sentiment vs Virality Score
        fig_sent_virality = px.scatter(df, x='sentiment_polarity', y='virality_score', color='is_trending',
                                      labels={'sentiment_polarity': 'Sentiment Polarity', 'virality_score': 'Virality Score', 'is_trending': 'Trending'},
                                      title='Sentiment Polarity vs Virality Score',
                                      color_discrete_map={True: 'orange', False: 'blue'})
        st.plotly_chart(fig_sent_virality, use_container_width=True)
        st.markdown("""
- Trending videos reach higher virality scores regardless of sentiment polarity.
- Peak engagement is centered around neutral sentiment, but high engagement outliers occur at all sentiment levels.
""")

        st.subheader("3. Named Entity Recognition (NER)")
        st.write("NER extracts names, brands, and events from video descriptions. Entities with high 'lift' in trending videos are potential emerging trends.")
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
            st.markdown("**Top Emerging Entities in Trending Videos (by Lift > 1.5):**")
            st.table({
                'Entity': [e[0] for e in emerging_entities],
                'Lift': [f"{e[1]:.2f}" for e in emerging_entities],
                'Trending Mentions': [e[2] for e in emerging_entities],
                'Interpretation': [
                    "Trending in niche or non-English content" if e[0].lower() == "comedia" else
                    "Possibly linked to academic season/news" if e[0].lower() == "harvard" else
                    "Spanish-language content trend" if e[0].lower() == "que" else
                    "Possibly related to world events or awareness" if e[0].lower() == "un" else
                    "Emerging or unique keyword in trending videos" for e in emerging_entities
                ]
            })
            st.markdown("""
**About Lift:**
- **Lift** measures how much more likely an entity is to appear in trending videos compared to non-trending ones.
- A lift value > 1.5 means the entity is at least 50% more common in trending content, making it a strong candidate for early trend detection.
- Entities with high lift and sufficient trending mentions are excellent signals for emerging trends.
""")
        else:
            st.info("No significant emerging entities found or NER data missing.")
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
                                labels={'value': 'Mentions', 'date_group': 'Date'},
                                title='Top 5 Entity Mention Volume Over Time')
            st.plotly_chart(fig_entity, use_container_width=True)
        else:
            st.info("No entity time series data available.")
        st.markdown("""
**Entity Trend Insights:**
- Spikes in entity mentions often signal viral or seasonal trends.
- Trending videos include more unique or emerging keywords than non-trending ones.
""")
    except Exception as e:
        st.error(f"Could not load NLP trend data or plot: {e}")

#------------------------------------------------------

elif selected_page == "Part 5: Predictive Model Development":
    st.header("Part 5: Predictive Model Development")
    st.write("Build machine learning models to predict which videos are likely to trend using various algorithms and feature engineering techniques.")
    
    # Model Overview
    st.subheader("🤖 Machine Learning Models Implemented")
    model_info = """
    | Model | Description | Key Features |
    |-------|-------------|--------------|
    | **Logistic Regression** | Linear classifier with balanced class weights | Fast, interpretable baseline model |
    | **Random Forest** | Ensemble of decision trees | Handles feature interactions, class imbalance |
    | **XGBoost** | Gradient boosting classifier | Advanced ensemble method, handles complex patterns |
    """
    st.markdown(model_info)
    
    # Feature Engineering Section
    st.subheader("🔧 Feature Engineering")
    st.write("**Virality Score Calculation:**")
    st.code("""
    virality_score = plays + (1 - corr_views_likes) * likes + 
                    (1 - corr_views_comments) * comments + 
                    (1 - corr_views_shares) * shares
    """, language='python')
    
    st.write("**Feature Categories:**")
    feature_categories = """
    - **Numerical Features:** Time (hour), hashtag count, description length, sentiment polarity, author metrics
    - **Categorical Features:** Day of week, weekend indicator, time period, author
    - **Text Features:** TF-IDF vectors from video descriptions (100 features)
    - **Author Historical Features:** Average virality, max virality, total videos, viral ratio
    """
    st.markdown(feature_categories)
      # Model Performance Section
    st.subheader("📊 Model Performance Comparison")
    
    # Display actual results from question4.md
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Validation AUC': [0.643, 0.549, 0.604],
        'Test AUC': [0.653, 0.545, 0.598],
        'Features Used': ['All features', 'All features', 'All features']
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    st.success("🏆 **Best Model:** Logistic Regression with AUC = 0.643")      # Display ROC Curve and Confusion Matrix side by side
    st.subheader("📈 Model Performance Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ROC Curve Analysis**")
        try:
            st.image("data/graph/question4-roc-curve-logistic.png", 
                    caption="ROC Curve - Logistic Regression (Test Set)", 
                    use_container_width=True)
        except Exception as e:
            st.error(f"Could not load ROC curve image: {e}")
    
    with col2:
        st.markdown("**Confusion Matrix**")
        try:
            st.image("data/graph/question4-confusion-matrix-logistic.png", 
                    caption="Confusion Matrix - Logistic Regression (Test Set)", 
                    use_container_width=True)
        except Exception as e:
            st.error(f"Could not load confusion matrix image: {e}")
      # Classification Report Results
    st.subheader("📋 Classification Report Results")
    
    # Create classification report as a table
    classification_data = {
        'Class': ['Non-trending (0)', 'Trending (1)', '', 'Accuracy', 'Macro avg', 'Weighted avg'],
        'Precision': [0.90, 0.21, '', '', 0.55, 0.81],
        'Recall': [0.70, 0.51, '', '', 0.60, 0.67],
        'F1-Score': [0.79, 0.29, '', 0.67, 0.54, 0.72],
        'Support': [626, 97, '', 723, 723, 723]
    }
    
    classification_df = pd.DataFrame(classification_data)
    st.dataframe(classification_df, use_container_width=True)
      # Time Series Analysis Section
    st.subheader("📈 Time Series Analysis & Forecasting")
    st.write("**ARIMA Model for Trend Prediction:**")
    
    arima_info = """
    - **Model:** ARIMA(7, 1, 2) with seasonal component
    - **Purpose:** Predict 30-day virality score trends
    - **Features:** Captures weekly seasonality and trend patterns
    - **Output:** Daily virality score forecasts with confidence intervals
    """
    st.markdown(arima_info)      # Display Time Series Analysis side by side
    st.subheader("📊 Time Series Analysis & Forecasting")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**7-Day Rolling Average Virality Score**")
        try:
            st.image("data/graph/question4-7dayrollingaverage-virality-score.png", 
                    caption="7-Day Rolling Average of Virality Scores Over Time", 
                    use_container_width=True)
        except Exception as e:
            st.error(f"Could not load rolling average image: {e}")
    
    with col4:
        st.markdown("**30-Day ARIMA Forecast**")
        try:
            st.image("data/graph/question4-30day-virality-score-forecast.png", 
                    caption="30-Day Virality Score Forecast with Confidence Intervals", 
                    use_container_width=True)
        except Exception as e:
            st.error(f"Could not load forecast image: {e}")
      # ARIMA Model Summary
    st.subheader("📈 ARIMA Model Summary")
    arima_summary = """
    **Model:** ARIMA(7, 1, 2)
    - **Observations:** 40 data points
    - **Log Likelihood:** 59.460
    - **AIC:** -98.919
    - **Significant Parameters:** ar.L6 coefficient (-0.5026) is statistically significant (p=0.005)
    """
    st.markdown(arima_summary)

#------------------------------------------------------

elif selected_page == "Part 6: AI-Driven Trend Forecasting":
    st.header("Part 6: AI-Driven Trend Forecasting")
    st.write("|   Model : Random Forest Classifier")

    # ROC CURVE - Side by Side
    st.header("📈 ROC CURVE ANALYSIS")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Next 24 Hours**")
        try:
            st.image("data/graph/ROC-24HOURS.png", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load 24-hour ROC image: {e}")

    with col2:
        st.markdown("**Next 7 Days**")
        try: 
            st.image("data/graph/ROC-7DAYS.png", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load 7-day ROC image: {e}")

    # VIRAL HASHTAG PREDICTION - Side by Side
    st.header("🏷️ VIRAL HASHTAG PREDICTION")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Next 24 Hours**")
        try:
            st.image("data/graph/HASHTAG-24.png", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load 24-hour hashtag image: {e}")

    with col4:
        st.markdown("**Next 7 Days**")
        try:
            st.image("data/graph/HASHTAG-7.png", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load 7-day hashtag image: {e}")

    # VIRAL CATEGORY PREDICTION - Side by Side
    st.header("📊 VIRAL CATEGORY PREDICTION")
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("**Next 24 Hours**")
        try:
            st.image("data/graph/PIE-24HOURS.png", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load 24-hour category image: {e}")

    with col6:
        st.markdown("**Next 7 Days**")
        try:
            st.image("data/graph/PIE-7DAYS.png", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load 7-day category image: {e}")

    # VIRAL ACCOUNT PREDICTION - Side by Side
    st.header("👤 VIRAL ACCOUNT PREDICTION")
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown("**Next 24 Hours**")
        try:
            df = pd.read_csv("data/processed/top_viral_24h.csv")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load 24-hour viral accounts data: {e}")

    with col8:
        st.markdown("**Next 7 Days**")
        try:
            df = pd.read_csv("data/processed/top_viral_7d.csv")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load 7-day viral accounts data: {e}")


#------------------------------------------------------

elif selected_page == "Part 7: Visualization":
    st.header("Part 7: Visualization")
    st.write("Visualize the results and findings.")

    # Load processed data
    data_path = 'data/processed/tiktok_processed_with_nlp_features.csv'
    df = pd.read_csv(data_path)

    # Preprocessing
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    df['date'] = df['create_time'].dt.date
    df.columns = df.columns.str.strip()

    # === Top Entities Over Time ===
    st.subheader('Top Extracted Entities Over Time')
    TOP_N = st.slider("Select number of top entities to display:", 3, 20, 5, key="entities_slider")

    if 'create_time' in df.columns and 'extracted_entities' in df.columns:
        df['extracted_entities'] = df['extracted_entities'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        ENTITY_LABELS_MAP = {
            "PERSON": "Person", "ORG": "Organization", "GPE": "Country/City/State", "LOC": "Location",
            "PRODUCT": "Product", "EVENT": "Event", "WORK_OF_ART": "Work of Art", "LAW": "Law",
            "LANGUAGE": "Language", "DATE": "Date", "TIME": "Time", "PERCENT": "Percent",
            "MONEY": "Money", "QUANTITY": "Quantity", "ORDINAL": "Ordinal", "CARDINAL": "Cardinal",
            "NORP": "Nationality/Religious/Political Group", "FAC": "Facility",
        }

        all_labels = set()
        for ents in df['extracted_entities']:
            if isinstance(ents, list):
                for ent in ents:
                    if isinstance(ent, (list, tuple)) and len(ent) == 2:
                        all_labels.add(ent[1])
        all_labels = sorted(all_labels)
        label_display_map = {ENTITY_LABELS_MAP.get(code, code): code for code in all_labels}
        display_labels = list(label_display_map.keys())

        selected_display_labels = st.multiselect(
            "Filter by entity category (label):",
            options=display_labels,
            default=display_labels,
            key="entity_labels"
        )
        selected_labels = [label_display_map[d] for d in selected_display_labels]

        entity_df = df.explode('extracted_entities')
        entity_df = entity_df.dropna(subset=['extracted_entities'])
        entity_df[['entity', 'label']] = entity_df['extracted_entities'].apply(
            lambda x: pd.Series(x) if isinstance(x, (list, tuple)) and len(x) == 2 else pd.Series([None, None])
        )
        entity_df = entity_df[entity_df['label'].isin(selected_labels)]

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
            title=f"Top {TOP_N} Mentioned Entities Over Time",
            labels={'mentions': 'Mentions', 'date': 'Date'}
        )
        st.plotly_chart(fig)
        st.caption("Shows frequency of top N named entities (topics) extracted per day, filtered by category.")
    else:
        st.warning("The required columns 'create_time' and 'extracted_entities' are missing from the dataset.")

    # === Timing vs Engagement ===
    st.subheader('Post Timing vs Engagement')
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
                y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                colorscale='YlGnBu',
                zsmooth='best',
                colorbar=dict(title="Engagement Rate Per Play")
            )
        )

        fig.update_layout(
            title="Smooth Heatmap: Engagement Rate by Post Timing",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            autosize=True
        )

        st.plotly_chart(fig)

    # === Sentiment vs Virality ===
    st.subheader('Sentiment vs Normalized Virality (Polarity ≠ 0)')
    if {'sentiment_polarity', 'sentiment_subjectivity', 'virality_score_normalized'}.issubset(df.columns):
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
            title='Sentiment vs Virality (Excluding Neutral Polarity)',
            labels={
                'sentiment_polarity': 'Polarity (Negative → Positive)',
                'virality_score_normalized': 'Normalized Virality',
                'sentiment_subjectivity': 'Subjectivity (Objective → Subjective)'
            },
            hover_data=['video_id'] if 'video_id' in df.columns else None
        )
        fig.update_layout(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[0, 0.5])
        )

        st.plotly_chart(fig)
        st.caption('Neutral-toned (polarity = 0) posts are excluded for clarity.')

    # === Trending Hashtags Monitor ===
    st.subheader('📈 Real-Time Trending Hashtag Monitor')
    if 'create_time' in df.columns and 'hashtag_list_clean' in df.columns:
        df['hashtag_list_clean'] = df['hashtag_list_clean'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        current_time = df['create_time'].max()
        time_window_days = st.slider("Time window (days)", 7, 30, 7, key="hashtag_slider")

        recent_df = df[df['create_time'] > (current_time - pd.Timedelta(days=time_window_days))]

        exploded = recent_df.explode('hashtag_list_clean')
        exploded = exploded.dropna(subset=['hashtag_list_clean'])
        exploded = exploded[
            exploded['hashtag_list_clean'].notna() &
            (exploded['hashtag_list_clean'].str.strip() != '')
        ]        
        top_hashtags = (
            exploded['hashtag_list_clean']
            .value_counts()
            .nlargest(10)
            .reset_index()
        )
        
        # Fix column names for newer pandas versions
        if 'hashtag_list_clean' in top_hashtags.columns:
            top_hashtags.columns = ['Hashtag', 'Mentions']
        elif 'index' in top_hashtags.columns:
            top_hashtags = top_hashtags.rename(columns={'index': 'Hashtag', 'hashtag_list_clean': 'Mentions'})
        else:
            # For newer pandas versions that use 'count' as default column name
            top_hashtags.columns = ['Hashtag', 'Mentions']

        fig = px.bar(
            top_hashtags,
            x='Mentions',
            y='Hashtag',
            title=f"Top Trending Hashtags (Last {time_window_days} Days)",
            orientation='h'
        )

        fig.update_layout(
            xaxis_title='Count',
            yaxis_title='Hashtag'
        )

        st.plotly_chart(fig)
        st.caption(f"Data filtered from: {current_time - pd.Timedelta(days=time_window_days):%Y-%m-%d %H:%M} to {current_time:%Y-%m-%d %H:%M}")
    else:
        st.warning("Required columns 'create_time' and 'hashtag_list_clean' not found.")

    # Optional: show full data
    with st.expander("Show Raw Data"):
        st.dataframe(df)


#------------------------------------------------------

elif selected_page == "Part 8 & 9: Summary":
    st.header("Part 8 & 9: Summary")
    st.write("Summarize the project and findings.")

    st.subheader("📌 Project Title:")
    st.markdown("**AI-Powered Trend Prediction on TikTok: Leveraging Data Science for Social Media Insights**")

    st.subheader("🎯 Project Objective:")
    st.write("To develop an AI-powered system that analyzes TikTok video data to identify, understand, and predict trends and viral content patterns.")

    st.subheader("📝 8. Insights Gained from the Project")
    st.markdown("""
    - **AI Application for Trend Forecasting:**
        - Predicts short-term virality using metadata, sentiment, and hashtags.
        - Helps brands, marketers, and influencers optimize post timing, content tone, and hashtag use.
    - **Recommendations:**
        - **Creators:** Post during midnight–5 AM, use 5–12 hashtags, and create emotionally engaging content.
        - **Brands:** Monitor high-lift trending entities and align messaging with top categories like Sports and DIY.
        - **Strategists:** Use model outputs (probability scores, forecasts) for proactive campaign planning.
    """)

    st.subheader("🚧 9. Challenges and Future Work")
    st.markdown("""
    - **Challenges:**
        - **Data Limitations:** API restrictions limited access to follower counts and full metadata.
        - **Imbalanced Classes:** Viral content is rare, requiring oversampling (e.g., SMOTE).
        - **External Influences:** Virality is also affected by real-world events, which are hard to capture.
    - **Future Work:**
        - **Real-Time Forecasting:** Integrate live pipelines for continuous trend monitoring.
        - **Deeper NLP:** Use transformer models like BERT for better semantic understanding.
        - **Cross-Platform Data:** Combine with external trend sources (e.g., Google Trends, Twitter).
        - **User-Level Trends:** Model creator behavior over time for personalized virality predictions.
    """)
