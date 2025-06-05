import streamlit as st
import pandas as pd
import plotly.express as px

# Sidebar navigation
tabs = st.sidebar.radio('Navigation', ['Trends', 'Engagement', 'Model Output', 'Recommendations'])

# Load processed data (update path as needed)
data_path = 'data/processed/tiktok_processed_with_nlp_features.csv'
df = pd.read_csv(data_path)

# Ensure all columns are accessible directly (no leading/trailing spaces, correct dtypes)
df.columns = df.columns.str.strip()

st.title('TikTok Trends & Virality Dashboard')

if tabs == 'Trends':
    st.header('Trend Evolution Over Time')
    if 'date' in df.columns:
        fig = px.line(df, x='date', y='video_views', title='Views Over Time')
        st.plotly_chart(fig)
    st.write('Explore how TikTok trends evolve over time.')

elif tabs == 'Engagement':
    
    st.header('Engagement Analysis')
    if 'likes' in df.columns and 'shares' in df.columns:
        color_col = 'primary_hashtag_category' if 'primary_hashtag_category' in df.columns else None
        fig = px.scatter(
            df, x='likes', y='shares', color=color_col,
            title='Likes vs Shares by Hashtag Category' if color_col else 'Likes vs Shares'
        )
        st.plotly_chart(fig)
    st.write('Visualize engagement metrics and their impact on virality.')

elif tabs == 'Model Output':
    st.header('Predictive Model Output')
    if 'predicted_virality' in df.columns:
        fig = px.histogram(df, x='predicted_virality', title='Predicted Virality Distribution')
        st.plotly_chart(fig)
    st.write('See how the AI model predicts TikTok virality.')

elif tabs == 'Recommendations':
    st.header('Recommendations for TikTok Strategy')
    st.markdown('''
    - Use trending hashtags and positive sentiment to boost reach.
    - Post at optimal times based on trend analysis.
    - Focus on content types that historically perform well.
    - Leverage AI predictions to optimize future posts.
    ''')
    st.write('These recommendations are based on the analysis and model insights.')

# Optionally, display the dataframe for user reference
with st.expander('Show DataFrame'):
    st.dataframe(df)
