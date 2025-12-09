"""
VOC Insight Engine - Voice of Customer Analytics
Professional Streamlit application for analyzing customer reviews with AI-powered insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
from utils import generate_correlated_reviews, calculate_nps

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="VOC Insight Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with modern color scheme - Teal/Blue/Emerald Professional Palette
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main content background - Dark theme */
.main {
    background-color: #0f172a !important;
}

.stApp {
    background-color: #0f172a !important;
}

/* Sidebar - Professional Dark Theme */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    color: #f1f5f9 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}

section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stDateInput label {
    color: #cbd5e1 !important;
}

/* Sidebar Metric Cards - Professional Teal/Blue Gradient */
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%) !important;
    border: none !important;
    padding: 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
    margin-bottom: 15px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4) !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] label {
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-top: 5px !important;
}

/* Main Content Metric Cards - Professional Color Palette */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%) !important;
    border: none !important;
    padding: 25px 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.25) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.35) !important;
}

div[data-testid="stMetric"] > div {
    width: 100% !important;
}

div[data-testid="stMetric"] label {
    color: rgba(255, 255, 255, 0.95) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-top: 8px !important;
}

/* Alternative metric card colors - Professional palette */
div[data-testid="stMetric"]:nth-child(1) {
    background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%) !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.25) !important;
}

div[data-testid="stMetric"]:nth-child(1):hover {
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.35) !important;
}

div[data-testid="stMetric"]:nth-child(2) {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
}

div[data-testid="stMetric"]:nth-child(2):hover {
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35) !important;
}

div[data-testid="stMetric"]:nth-child(3) {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
}

div[data-testid="stMetric"]:nth-child(3):hover {
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35) !important;
}

div[data-testid="stMetric"]:nth-child(4) {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.25) !important;
}

div[data-testid="stMetric"]:nth-child(4):hover {
    box-shadow: 0 8px 25px rgba(245, 158, 11, 0.35) !important;
}

/* Tabs - Dark Theme Matching Sidebar */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #1e293b !important;
    padding: 8px;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    background-color: #334155 !important;
    border-radius: 8px;
    border: 2px solid transparent;
    color: #cbd5e1 !important;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #475569 !important;
    color: #f1f5f9 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%) !important;
    color: white !important;
    border: 2px solid transparent !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
}

/* Button Styling - Professional Teal/Blue */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
}

/* Headers - White text for dark theme */
h1, h2, h3, h4, h5, h6 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}

/* All text content - White/Light colors */
p, li, td, th, div, span {
    color: #e2e8f0 !important;
}

/* Markdown content - Light text */
.stMarkdown {
    color: #e2e8f0 !important;
    line-height: 1.7;
}

.stMarkdown p {
    color: #e2e8f0 !important;
}

.stMarkdown strong {
    color: #f1f5f9 !important;
}

.stMarkdown em {
    color: #cbd5e1 !important;
}

.stMarkdown ul, .stMarkdown ol {
    color: #e2e8f0 !important;
}

.stMarkdown li {
    color: #e2e8f0 !important;
}

/* Tables - Dark theme */
.stMarkdown table {
    border-radius: 8px;
    overflow: hidden;
    background-color: #1e293b !important;
}

.stMarkdown table th {
    background-color: #334155 !important;
    color: #f1f5f9 !important;
    padding: 12px !important;
}

.stMarkdown table td {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    padding: 10px !important;
    border-color: #334155 !important;
}

/* Code blocks - Professional Dark Theme */
.stMarkdown code {
    background-color: #1e293b !important;
    color: #10b981 !important;
    padding: 4px 8px !important;
    border-radius: 4px !important;
    border: 1px solid #334155 !important;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
    font-size: 0.9em !important;
}

/* Code blocks (multi-line) - Dark theme */
.stMarkdown pre {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    overflow-x: auto !important;
}

.stMarkdown pre code {
    background-color: transparent !important;
    color: #e2e8f0 !important;
    border: none !important;
    padding: 0 !important;
    font-size: 0.9em !important;
    line-height: 1.6 !important;
}

/* Syntax highlighting for code blocks */
.stMarkdown pre code {
    color: #e2e8f0 !important;
}

/* Text Inputs and Selectboxes - Dark theme */
.stTextInput > div > div > input,
.stSelectbox > div > div > select {
    border-radius: 8px;
    border: 2px solid #334155 !important;
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    transition: border-color 0.3s ease;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
}

/* Date Input - Dark theme */
.stDateInput > div > div > input {
    border-radius: 8px;
    border: 2px solid #334155 !important;
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
}

/* Dataframe Styling - Dark theme */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
    background-color: #1e293b !important;
}

.dataframe thead th {
    background-color: #334155 !important;
    color: #f1f5f9 !important;
}

.dataframe tbody td {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
}

/* Divider */
hr {
    border: none;
    border-top: 2px solid #334155 !important;
    margin: 2rem 0;
}

/* Info boxes - Dark theme */
.stInfo {
    background-color: #1e3a5f !important;
    border-left: 4px solid #0ea5e9 !important;
    border-radius: 8px;
    color: #e2e8f0 !important;
}

.stWarning {
    background-color: #3d2817 !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 8px;
    color: #e2e8f0 !important;
}

.stError {
    background-color: #3d1f1f !important;
    border-left: 4px solid #ef4444 !important;
    border-radius: 8px;
    color: #e2e8f0 !important;
}

/* Caption text */
.stCaption {
    color: #94a3b8 !important;
}

/* Scrollbar Styling - Dark theme */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #1e293b;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #64748b;
}

/* Multiselect - Dark theme */
.stMultiSelect > div > div {
    background-color: #1e293b !important;
    border: 2px solid #334155 !important;
    color: #f1f5f9 !important;
}

.stMultiSelect > div > div:focus {
    border-color: #0ea5e9 !important;
}

/* Selectbox dropdown - Dark theme */
div[data-baseweb="select"] {
    background-color: #1e293b !important;
}

div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
}

/* Title and subtitle */
h1 {
    color: #f1f5f9 !important;
}

/* Streamlit default text elements */
.element-container {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Generate and cache the review data with fixed seed for reproducibility."""
    df = generate_correlated_reviews(n_reviews=200, months_back=12, seed=42)
    if df is None or len(df) == 0:
        raise ValueError("Failed to generate review data")
    return df


@st.cache_data(ttl=600)  # Cache AI summaries for 10 minutes
def generate_ai_summary(negative_reviews_text):
    """Generate executive summary using Google Gemini API."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            return """⚠️ **API Key Not Configured**

Please configure your Google Gemini API key:
1. Get your API key from: https://makersuite.google.com/app/apikey
2. Edit `.streamlit/secrets.toml` and add: `GEMINI_API_KEY = "your-key"`
3. Restart the Streamlit app"""
        
        genai.configure(api_key=api_key)
        
        # Try to list and select available models
        try:
            available_models = [m.name for m in genai.list_models() 
                              if 'generateContent' in m.supported_generation_methods]
            gemini_models = [m for m in available_models if 'gemini' in m.lower()]
            
            if not gemini_models:
                return "⚠️ No Gemini models available for your API key."
            
            # Prefer flash or pro models
            preferred = [m for m in gemini_models if 'flash' in m.lower() or 'pro' in m.lower()]
            model_name = preferred[0] if preferred else gemini_models[0]
            
            if model_name.startswith('models/'):
                model_name = model_name.replace('models/', '')
            
            model = genai.GenerativeModel(model_name)
        except Exception:
            # Fallback to common model names
            for name in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
                try:
                    model = genai.GenerativeModel(name)
                    break
                except:
                    continue
            else:
                return "⚠️ Could not connect to Gemini API. Please check your API key."
        
        prompt = """You are a Product Manager. Summarize the following negative reviews into 3 distinct bullet points highlighting the specific technical flaws users are reporting. Be concise and actionable.

Negative Reviews:
""" + negative_reviews_text
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        if "400" in error_msg or "API_KEY" in error_msg.upper() or "invalid" in error_msg.lower():
            return f"⚠️ **Invalid API Key** - Please update your API key in `.streamlit/secrets.toml`\n\nError: {error_msg}"
        return f"⚠️ **Error:** {error_msg}"


def filter_data(df, date_range, product_filter):
    """Apply filters to the dataframe."""
    filtered_df = df.copy()
    
    # Date range filter
    if date_range:
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range[0], date_range[1]
        elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
            start_date = end_date = date_range[0]
        else:
            start_date = end_date = date_range
        
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) &
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Product filter
    if product_filter and product_filter != 'All':
        filtered_df = filtered_df[filtered_df['product'] == product_filter]
    
    return filtered_df


def render_metrics(filtered_df):
    """Calculate and return metrics dictionary."""
    n = len(filtered_df)
    return {
        'total': n,
        'avg_sentiment': filtered_df['sentiment_score'].mean() if n > 0 else 0.0,
        'nps': calculate_nps(filtered_df) if n > 0 else 0,
        'negative': len(filtered_df[filtered_df['sentiment_category'] == 'Negative']) if n > 0 else 0
    }


def main():
    # Title
    st.title("📊 VOC Insight Engine")
    st.markdown("**Voice of Customer Analytics Dashboard** — Analyze customer feedback with AI-powered insights")
    
    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Initialize session state for filters
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Date range filter
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter"
        )
        
        # Product filter
        products = ['All'] + sorted(df['product'].unique().tolist())
        product_filter = st.selectbox("📦 Product Category", products, key="product_filter")
        
        st.divider()
        
        # Quick Stats in Sidebar
        st.subheader("📈 Quick Stats")
        
        # Apply filters
        filtered_df = filter_data(df, date_range, product_filter)
        metrics = render_metrics(filtered_df)
        
        # Display sidebar metrics using native st.metric
        st.metric(label="Total Reviews", value=metrics['total'])
        st.metric(label="Avg Sentiment", value=f"{metrics['avg_sentiment']:.2f}")
        st.metric(label="NPS Score", value=f"{metrics['nps']:.0f}")
    
    # Main content - Tabs (REORDERED: About first, Dashboard second, Deep Dive last)
    tab1, tab2, tab3 = st.tabs(["ℹ️ About", "📊 Dashboard", "🔍 Deep Dive"])
    
    # Apply filters for main content
    filtered_df = filter_data(df, date_range, product_filter)
    metrics = render_metrics(filtered_df)
    
    # Tab 1: About (First)
    with tab1:
        st.header("About VOC Insight Engine")
        
        st.markdown("""
### Project Overview

The **VOC Insight Engine** is a professional analytics platform for analyzing customer feedback 
with AI-powered insights using Google Gemini.

### Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Sentiment Analysis | VADER |
| AI/ML | Google Gemini 1.5 Flash |
| Visualization | Plotly Express |
| Data Generation | Faker |

### Key Features

1. **Correlated Data Generation** — Synthetic reviews with realistic defect patterns
2. **VADER Sentiment Analysis** — Compound sentiment scores
3. **AI-Powered Insights** — Google Gemini analyzes negative feedback
4. **Interactive Visualizations** — Trend lines, donut charts, bar charts
5. **Advanced Filtering** — Date range, product, sentiment, text search

### Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure your Gemini API key in `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

### Architecture

```
┌─────────────────┐
│   Streamlit UI  │
│  (app.py)       │
└────────┬────────┘
         │
┌────────▼────────┐
│  Data Layer     │
│  (utils.py)     │
└────────┬────────┘
         │
┌────────▼────────┐
│  AI Engine      │
│  (Gemini API)   │
└─────────────────┘
```

### Data Flow

1. **Data Generation:** `utils.py` generates 200 correlated reviews spanning 12 months
2. **Sentiment Analysis:** VADER calculates sentiment scores for each review
3. **Filtering:** User applies filters via sidebar and deep dive tabs
4. **AI Analysis:** Negative reviews are sent to Gemini API for executive summary
5. **Visualization:** Plotly renders interactive charts based on filtered data
        """)
    
    # Tab 2: Dashboard (Second)
    with tab2:
        # Dashboard Overview Header
        st.header("Dashboard Overview")
        
        # KPI Metrics Row - Using columns with native st.metric
        st.subheader("Key Performance Indicators")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.metric(
                label="📝 Total Reviews",
                value=metrics['total']
            )
        
        with kpi_col2:
            st.metric(
                label="😊 Avg Sentiment",
                value=f"{metrics['avg_sentiment']:.2f}"
            )
        
        with kpi_col3:
            st.metric(
                label="📊 NPS Score",
                value=f"{metrics['nps']:.0f}"
            )
        
        with kpi_col4:
            st.metric(
                label="👎 Negative Reviews",
                value=metrics['negative']
            )
        
        st.divider()
        
        # AI Executive Summary Section
        st.subheader("🤖 AI-Powered Executive Summary")
        st.caption("Analyze negative feedback patterns using Google Gemini AI")
        
        if st.button("⚡ Analyze Negative Feedback with AI", type="primary", key="ai_button"):
            negative_reviews = filtered_df[filtered_df['sentiment_category'] == 'Negative']
            
            if len(negative_reviews) > 0:
                with st.spinner("🔄 Generating AI summary..."):
                    negative_text = "\n\n".join(negative_reviews['review_text'].tolist())
                    summary = generate_ai_summary(negative_text)
                    st.markdown("### 📝 Executive Summary")
                    st.markdown(summary)
            else:
                st.warning("⚠️ No negative reviews found in the filtered dataset.")
        
        st.divider()
        
        # Visualizations
        st.subheader("📊 Analytics Visualizations")
        
        if len(filtered_df) > 0:
            # Chart 1: Sentiment Trend
            st.markdown("#### 📈 Sentiment Trend (7-Day Rolling Average)")
            
            daily_sentiment = filtered_df.groupby(
                filtered_df['date'].dt.date
            )['sentiment_score'].mean().reset_index()
            daily_sentiment.columns = ['date', 'avg_sentiment']
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            daily_sentiment = daily_sentiment.sort_values('date')
            daily_sentiment['rolling_avg'] = daily_sentiment['avg_sentiment'].rolling(
                window=7, min_periods=1
            ).mean()
            
            fig_line = px.line(
                daily_sentiment,
                x='date',
                y='rolling_avg',
                labels={'rolling_avg': 'Sentiment Score', 'date': 'Date'},
                color_discrete_sequence=['#0ea5e9']
            )
            fig_line.update_layout(
                hovermode='x unified',
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Charts Row: Donut + Bar
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("#### 🍩 Sentiment Distribution")
                
                sentiment_counts = filtered_df['sentiment_category'].value_counts()
                colors = {'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#ef4444'}
                
                fig_donut = go.Figure(data=[go.Pie(
                    labels=sentiment_counts.index.tolist(),
                    values=sentiment_counts.values.tolist(),
                    hole=0.4,
                    marker_colors=[colors.get(c, '#64748b') for c in sentiment_counts.index]
                )])
                fig_donut.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            
            with chart_col2:
                st.markdown("#### 📦 Sentiment by Product")
                
                product_sentiment = filtered_df.groupby('product')['sentiment_score'].mean().reset_index()
                product_sentiment = product_sentiment.sort_values('sentiment_score', ascending=True)
                
                fig_bar = px.bar(
                    product_sentiment,
                    x='sentiment_score',
                    y='product',
                    orientation='h',
                    labels={'sentiment_score': 'Avg Sentiment', 'product': 'Product'},
                    color='sentiment_score',
                    color_continuous_scale='Blues'
                )
                fig_bar.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("📭 No data available for the selected filters.")
    
    # Tab 3: Deep Dive (Last)
    with tab3:
        st.header("Deep Dive: Raw Data Analysis")
        
        # Filters for deep dive
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=['Positive', 'Neutral', 'Negative'],
                default=['Positive', 'Neutral', 'Negative'],
                key="sentiment_multiselect"
            )
        
        with filter_col2:
            search_text = st.text_input("🔍 Search in reviews", "", key="search_input")
        
        # Apply additional filters
        deep_dive_df = filtered_df.copy()
        
        if sentiment_filter:
            deep_dive_df = deep_dive_df[deep_dive_df['sentiment_category'].isin(sentiment_filter)]
        
        if search_text:
            deep_dive_df = deep_dive_df[
                deep_dive_df['review_text'].str.contains(search_text, case=False, na=False)
            ]
        
        st.markdown(f"**Showing {len(deep_dive_df)} reviews**")
        
        # Display data table
        if len(deep_dive_df) > 0:
            display_df = deep_dive_df[['date', 'product', 'review_text', 'sentiment_score', 'sentiment_category']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['sentiment_score'] = display_df['sentiment_score'].round(3)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500,
                hide_index=True
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"voc_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("📭 No reviews match the current filters.")


if __name__ == "__main__":
    main()
