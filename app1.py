import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt
import ast
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

# Sidebar navigation
tabs = st.sidebar.radio('Navigation', [
    'Top Entities Over Time',
    'Timing vs Engagement',
    'Sentiment vs Virality',
    'Trending Hashtags Monitor',
])

# Load processed data (update path as needed)
data_path = 'data/processed/tiktok_processed_with_nlp_features.csv'
df = pd.read_csv(data_path)

# Optional preprocessing
df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
df['date'] = df['create_time'].dt.date

# Ensure all columns are accessible directly (no leading/trailing spaces, correct dtypes)
df.columns = df.columns.str.strip()

st.title('TikTok Trends & Virality Dashboard')

# 1. Top Entities Over Time

if tabs == 'Top Entities Over Time':
    st.header('Top Extracted Entities Over Time')

    # Parameters
    TOP_N = st.slider("Select number of top entities to display:", 3, 20, 5)

    # Check required columns
    if 'create_time' in df.columns and 'extracted_entities' in df.columns:
        # Ensure datetime and extract date
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df['date'] = df['create_time'].dt.date

        # Convert stringified list to actual list (if needed)
        df['extracted_entities'] = df['extracted_entities'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Mapping from entity label codes to human-readable names
        ENTITY_LABELS_MAP = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Country/City/State",
            "LOC": "Location",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "WORK_OF_ART": "Work of Art",
            "LAW": "Law",
            "LANGUAGE": "Language",
            "DATE": "Date",
            "TIME": "Time",
            "PERCENT": "Percent",
            "MONEY": "Money",
            "QUANTITY": "Quantity",
            "ORDINAL": "Ordinal",
            "CARDINAL": "Cardinal",
            "NORP": "Nationality/Religious/Political Group",
            "FAC": "Facility",
        }

        # Flatten and extract all unique entity labels for filtering
        all_labels = set()
        for ents in df['extracted_entities']:
            if isinstance(ents, list):
                for ent in ents:
                    if isinstance(ent, (list, tuple)) and len(ent) == 2:
                        all_labels.add(ent[1])
        all_labels = sorted(all_labels)

        # Create mapping for multiselect: human-readable -> code
        label_display_map = {ENTITY_LABELS_MAP.get(code, code): code for code in all_labels}
        display_labels = list(label_display_map.keys())

        # Category filter (show human-readable, store code)
        selected_display_labels = st.multiselect(
            "Filter by entity category (label):",
            options=display_labels,
            default=display_labels
        )
        selected_labels = [label_display_map[d] for d in selected_display_labels]

        # Explode entity lists and filter by selected labels
        entity_df = df.explode('extracted_entities')
        # Parse entity and label
        entity_df = entity_df.dropna(subset=['extracted_entities'])
        entity_df[['entity', 'label']] = entity_df['extracted_entities'].apply(
            lambda x: pd.Series(x) if isinstance(x, (list, tuple)) and len(x) == 2 else pd.Series([None, None])
        )
        entity_df = entity_df[entity_df['label'].isin(selected_labels)]

        # Count top N entities overall (by entity text, not label)
        top_entities = entity_df['entity'].value_counts().nlargest(TOP_N).index.tolist()
        filtered_entity_df = entity_df[entity_df['entity'].isin(top_entities)]

        # Group by date and entity
        trend_df = (
            filtered_entity_df.groupby(['date', 'entity'])
            .size()
            .reset_index(name='mentions')
        )

        # Plot line chart
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


# 2. Timing vs Engagement

elif tabs == 'Timing vs Engagement':
    st.header('Post Timing vs Engagement')

    if 'create_hour' in df.columns and 'day_of_week' in df.columns and 'weighted_engagement_rate' in df.columns:

        # Create pivot table with weighted_engagement_rate
        pivot = df.pivot_table(
            values='engagement_rate_per_play_capped',
            index='day_of_week',
            columns='create_hour',
            aggfunc='mean'
        ).sort_index()

        # Plot smooth heatmap
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



# 3. Sentiment vs Virality
elif tabs == 'Sentiment vs Virality':
    st.header('Sentiment vs Normalized Virality (Polarity ≠ 0)')

    if {'sentiment_polarity', 'sentiment_subjectivity', 'virality_score_normalized'}.issubset(df.columns):
        # Filter out neutral polarity
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

# 4. Trending Hashtags Monitor
elif tabs == 'Trending Hashtags Monitor':
    st.header('📈 Real-Time Trending Hashtag Monitor')

    # Check necessary columns
    if 'create_time' in df.columns and 'hashtag_list_clean' in df.columns:
        # Convert datetime and evaluate lists
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')

        # Optional: convert stringified list to real list
        df['hashtag_list_clean'] = df['hashtag_list_clean'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Set current time and define sliding window
        current_time = df['create_time'].max()
        time_window_days = st.slider("Time window (days)", 7, 30, 7)

        # Filter rows within the sliding time window
        recent_df = df[df['create_time'] > (current_time - pd.Timedelta(hours=time_window_days))]

        # Explode hashtag list
        exploded = recent_df.explode('hashtag_list_clean')
        exploded = exploded.dropna(subset=['hashtag_list_clean'])
        
        # Drop blanks and NaNs
        exploded = exploded[
            exploded['hashtag_list_clean'].notna() &  # remove NaNs
            (exploded['hashtag_list_clean'].str.strip() != '')  # remove empty strings
        ]


        # Count top hashtags
        top_hashtags = (
            exploded['hashtag_list_clean']
            .value_counts()
            .nlargest(10)
            .reset_index()
            .rename(columns={'index': 'Hashtag', 'hashtag_list_clean': 'Mentions'})
        )

        # Plot
        fig = px.bar(
            top_hashtags,
            x='Mentions',
            y='count',
            title=f"Top Trending Hashtags (Last {time_window_days} Days)",
            labels={'Mentions': 'Count'}
            )
        fig.update_layout(
                xaxis_title='Hashtag',
                yaxis_title='Count',
            )
        

        st.plotly_chart(fig)
        st.caption(f"Data filtered from: {current_time - pd.Timedelta(hours=time_window_days):%Y-%m-%d %H:%M} to {current_time:%Y-%m-%d %H:%M}")
    else:
        st.warning("Required columns 'create_time' and 'hashtag_list_clean' not found.")


# Optionally, display the dataframe for user reference
with st.expander('Show DataFrame'):
    st.dataframe(df)
