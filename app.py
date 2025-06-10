import streamlit as st
import pandas as pd
import numpy as np

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
st.title("TikTok Data Analysis App")

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
    st.write("| posted_time & create_time --> create_time")
    st.write("| views & plays --> plays")
    df['create_time'] = df['create_time'].fillna(df['posted_time'])
    df['plays'] = df['plays'].fillna(df['views'])
    st.write("\nNull values in create_time:", df['create_time'].isnull().sum())
    st.write("Null values in plays:", df['plays'].isnull().sum())

    st.subheader("1.2 Null Value Handling")
    st.write("Description null value --> No Description")
    st.write("Hashtags null value --> empty list [  ]")

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
    st.write(" Total engagement = likes + comments + shares")
    st.write(" Engagement rate per play = total engagement / plays")

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
    st.write("Use NLP techniques to identify trends.")

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
    
    st.success("🏆 **Best Model:** Logistic Regression with AUC = 0.643")
      # Display ROC Curve
    st.subheader("📈 ROC Curve Analysis")
    try:
        st.image("data/graph/question4-roc-curve-logistic.png", 
                caption="ROC Curve - Logistic Regression (Test Set)", 
                use_container_width=True)
    except Exception as e:
        st.error(f"Could not load ROC curve image: {e}")
    
    # Display Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
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
    st.markdown(arima_info)
      # Display 7-Day Rolling Average
    st.subheader("📊 7-Day Rolling Average Virality Score")
    try:
        st.image("data/graph/question4-7dayrollingaverage-virality-score.png", 
                caption="7-Day Rolling Average of Virality Scores Over Time", 
                use_container_width=True)
    except Exception as e:
        st.error(f"Could not load rolling average image: {e}")
    
    # Display ARIMA Forecast
    st.subheader("🔮 30-Day ARIMA Forecast")
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

    st.header("ROC CURVE")

    st.subheader("Next 24 Hours")

    try:
        st.image("data/graph/ROC-24HOURS.png")
    except Exception as e:
        st.error(f"Could not load image: {e}")

    st.subheader("Next 7 Days")
    try: 
        st.image("data/graph/ROC-7DAYS.png")
    except Exception as e:
        st.error(f"Could not load image: {e}")



    st.header("VIRAL HASHTAG PREDICTION")

    st.subheader("Next 24 Hours")
    try:
        st.image("data/graph/HASHTAG-24.png")
    except Exception as e:
        st.error(f"Could not load image: {e}")

    st.subheader("Next 7 Days")
    try:
        st.image("data/graph/HASHTAG-7.png")
    except Exception as e:
        st.error(f"Could not load image: {e}")


    st.header("VIRAL CATEGORY PREDICTION")
    st.subheader("Next 24 Hours")

    try:
        st.image("data/graph/PIE-24HOURS.png")
    except Exception as e:
        st.error(f"Could not load image: {e}")

    st.subheader("Next 7 Days")
    try:
        st.image("data/graph/PIE-7DAYS.png")
    except Exception as e:
        st.error(f"Could not load image: {e}")


    st.header("VIRAL ACCOUNT PREDICTION")

    st.subheader("Next 24 Hours")
    try:
        df = pd.read_csv("data/processed/top_viral_24h.csv")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Could not load image: {e}")

    st.subheader("Next 7 Days")
    try:
        df = pd.read_csv("data/processed/top_viral_7d.csv")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Could not load image: {e}")


#------------------------------------------------------

elif selected_page == "Part 7: Visualization":
    st.header("Part 7: Visualization")
    st.write("Visualize the results and findings.")

#------------------------------------------------------

elif selected_page == "Part 8 & 9: Summary":
    st.header("Part 8 & 9: Summary")
    st.write("Summarize the project and findings.")