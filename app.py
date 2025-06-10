import streamlit as st
import pandas as pd

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
    st.write("Explore the data with visualizations and statistics.")

#------------------------------------------------------



elif selected_page == "Part 4: Trend Identification Using NLP":
    st.header("Part 4: Trend Identification Using NLP")
    st.write("Use NLP techniques to identify trends.")

#------------------------------------------------------

elif selected_page == "Part 5: Predictive Model Development":
    st.header("Part 5: Predictive Model Development")
    st.write("Develop predictive models based on the data.")

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