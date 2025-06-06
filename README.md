# TikTok Trend Analysis Project 🔥

This project collects and analyzes TikTok data to identify viral content patterns, predict trending videos, and extract meaningful insights using machine learning and NLP techniques.

## Table of Contents

- [Setup](#setup)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Predictive Model Development](#predictive-model-development)
- [Trend Identification Using NLP](#trend-identification-using-nlp)
- [AI-Driven Trend Forecasting](#ai-driven-trend-forecasting)
- [Visualization](#visualization)
- [Optional Enhancements](#optional-enhancements)
- [Project Structure](#project-structure)

## Setup

### Prerequisites

1. Python 3.8+ installed
2. Git (optional)

### Installation

1. Clone this repository or download the code

```bash
git clone [repository-url]
cd Tiktok_Py
```

2. Install required packages

```bash
pip install -r requirements.txt
```

3. **Important**: Create a `.env` file in the root directory with your RapidAPI key:

```
RAPIDAPI_KEY=your_rapidapi_key_here
```

To get a RapidAPI key:

1. Sign up at [RapidAPI](https://rapidapi.com/)
2. Subscribe to the [TikTok API](https://rapidapi.com/search/tiktok)
3. Copy your API key and paste it in the `.env` file

## Data Collection

### 🧠 Goal

Collect a comprehensive dataset with TikTok video-level data to identify viral content patterns.

### ✅ What We Do

- Use TikTok API via RapidAPI to extract trending TikTok posts
- Focus on rich metadata and engagement signals
- Store data in structured CSV format

### 📥 Data Collected

- `video_id`: Unique identifier for each video
- `description`: Video caption text
- `hashtags`: Tags used in the video
- `likes`: Number of likes
- `shares`: Number of shares
- `comments`: Number of comments
- `followers_count`: Creator's follower count
- `create_time`: When the video was posted
- `duration`: Video length
- `sound/music`: Audio used in the video
- `author`: Creator username
- `verified_status`: Whether the account is verified

### 🛠️ Tools Used

- TikTok API (via RapidAPI)
- Python requests library
- Pandas for data handling

### Usage

Run the data collection notebook to gather trending TikTok data:

```bash
jupyter notebook retrieve_data_v2.ipynb
```

## Data Preprocessing

### 🧠 Goal

Clean, standardize, and engineer new features for modeling.

### ✅ What We Do

- Clean text data (remove emojis, links, stopwords)
- Convert timestamps into usable time features
- Normalize engagement metrics to account for follower count
- Handle missing values and outliers
- Merge data from different collection runs

### 🛠️ Techniques

- Pandas and NumPy for data manipulation
- NLTK/spaCy for text processing
- Regular expressions for text cleaning
- StandardScaler/MinMaxScaler for feature normalization

### 🆕 New Features

- `engagement_rate` = (likes + comments + shares) / followers
- Time features: `hour`, `day_of_week`, `post_day_type` (weekday/weekend)
- Cleaned description and hashtags
- Optional: `word_count`, `hashtag_count`, `video_length_group` (short/medium/long)

### Usage

Run the data preprocessing notebook:

```bash
jupyter notebook merge_data_v2.ipynb
```

## Exploratory Data Analysis

### 🧠 Goal

Understand the behavior and factors that make videos go viral.

### ✅ What We Do

- Visualize engagement metrics across different times/days
- Analyze sentiment in relation to video popularity
- Explore relationships between hashtags, posting time, and video success
- Identify patterns in viral content creation across different accounts

### 🛠️ Techniques

- Visualization libraries: Seaborn, Matplotlib, Plotly
- Statistical analysis: correlation matrices, distribution plots
- Segmentation analysis by time, creator type, and content category

### 🆕 Features Created for Analysis

- `sentiment_score` (using VADER/TextBlob/BERT)
- Content categories (dance, fashion, comedy) using NLP techniques

## Predictive Model Development

### 🧠 Goal

Build ML models to predict which videos are likely to trend.

### ✅ What We Do

- Define target variable: `is_trending` (1 if engagement rate > threshold)
- Split data into training/testing sets
- Train and evaluate different models
- Feature engineering focused on metadata and NLP features

### 🛠️ Techniques

- Models: Logistic Regression, Random Forest, XGBoost, LightGBM, MLP
- NLP Features: TF-IDF or Sentence Embeddings from descriptions/hashtags
- Model evaluation: Accuracy, ROC AUC, Precision/Recall, Confusion Matrix

### 🆕 Features for Model

- `engagement_rate`
- Post time features
- Sentiment analysis
- Topic modeling
- Named entities
- Hashtag frequency analysis

## Trend Identification Using NLP

### 🧠 Goal

Use Natural Language Processing to uncover hidden or emerging themes in content.

### ✅ What We Do

- Apply topic modeling to discover common content themes
- Use Named Entity Recognition to extract trending names, brands, and places
- Compare sentiment to performance to identify emotional hooks

### 🛠️ Techniques

- LDA with Gensim/Sklearn for topic discovery
- spaCy or HuggingFace for Named Entity Recognition
- VADER/TextBlob for sentiment scoring

### 🆕 Features Created

- `topic`: Content categories (0-5 or more)
- `named_entities`: Extracted entities (e.g., "Taylor Swift", "Met Gala")
- `sentiment_label`: Positive/Neutral/Negative classification

## AI-Driven Trend Forecasting

### 🧠 Goal

Forecast which types of content, hashtags, or topics will become viral soon.

### ✅ What We Do

- Train models to predict "virality probability" for new content
- Add temporal dimension for time-based predictions
- Optimize for different forecasting windows (24hrs / 7 days)

### 🛠️ Techniques

- Time-aware ML (lag features, rolling averages)
- Classification (binary trend prediction) or Regression (engagement score)
- Feature Selection using RFE or SHAP values

### 🆕 Output

- Predicted probability of virality
- List of potentially trending hashtags for upcoming days
- Content category forecasts

## Visualization

### 🧠 Goal

Communicate insights, patterns, and model outputs effectively.

### ✅ What We Do

Build interactive dashboards showing:

- Engagement patterns over time
- Feature importance for prediction models
- Trend timelines and forecasts
- Content category analysis

### 🛠️ Tools

- Streamlit for interactive web applications
- Plotly for dynamic, interactive charts
- Matplotlib/Seaborn for static visualizations

### 📊 Example Visualizations

- Time series of top hashtags
- Heatmap of post timing vs. engagement
- Sentiment vs. virality scatterplots
- Real-time trending hashtag monitor

## Optional Enhancements

### 💡 Future Ideas

- Add clustering to segment users or videos by behavior
- Integrate Google Trends or Twitter data for external signals
- Explore multimodal models (combining video + text features)
- Real-time monitoring and alerting system for emerging trends
- Integration with content creation tools

## Project Structure

```
TikTok_Py/
├── .env                  # Environment variables (API keys)
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── retrieve_data.ipynb   # Initial data collection notebook
├── retrieve_data_v2.ipynb# Improved data collection process
├── merge_data.ipynb      # Initial data merging and processing
├── merge_data_v2.ipynb   # Enhanced data preprocessing
├── raw_data.csv          # Combined raw data
├── raw_data.xlsx         # Excel version of raw data
├── removeduplicate_data.csv # Cleaned data without duplicates
├── data/                 # Directory containing additional datasets
└── tiktok_trending*.csv  # Individual data collection files
```

---

## How to Run Streamlit

To launch the Streamlit app, run:

```bash
streamlit run app.py
```

### Troubleshooting: `'streamlit' is not recognized as an internal or external command`

If you get a "command not found" error, Streamlit may not be in your system `PATH`. To fix this temporarily in your current PowerShell session, run:

```powershell
$env:Path += ";$env:USERPROFILE\AppData\Roaming\Python\Python312\Scripts"
```

Now try running Streamlit again.

#### Make the Change Permanent

To avoid repeating this step, add the Scripts folder to your `PATH` permanently:

1. Search for **"Environment Variables"** in the Start menu and open **"Edit the system environment variables"**.
2. Click **"Environment Variables..."**.
3. Under **"User variables"**, select **Path** and click **Edit**.
4. Click **New** and add:
   ```
   %USERPROFILE%\AppData\Roaming\Python\Python312\Scripts
   ```
5. Click **OK** on all windows to save changes.
6. Restart your terminal for the changes to take effect.

**Note**: This project is for educational and research purposes only. Always ensure you comply with TikTok's Terms of Service when collecting and using data from their platform.
