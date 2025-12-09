# VOC Insight Engine

A professional-grade Streamlit application for Voice of Customer Analytics that analyzes customer reviews, visualizes sentiment trends, and generates AI-powered executive summaries using Google Gemini.

## 🌐 Live Application

**Access the live application:** [https://voice-of-customer-analytics.streamlit.app/](https://voice-of-customer-analytics.streamlit.app/)

## Features

- 📊 **Interactive Dashboard** with real-time sentiment analysis
- 🤖 **AI-Powered Insights** using Google Gemini 1.5 Flash
- 📈 **Advanced Visualizations** with Plotly (trends, distributions, comparisons)
- 🔍 **Deep Dive Analysis** with filtering and search capabilities
- 📥 **Data Export** functionality

## Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Sentiment Analysis:** VADER
- **AI/ML:** Google Gemini 1.5 Flash
- **Visualization:** Plotly Express
- **Data Generation:** Faker

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API key:
   - Create/edit `.streamlit/secrets.toml`
   - Add your Google Gemini API key:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     ```
   - Get your API key from: https://makersuite.google.com/app/apikey

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Dashboard Tab:** View KPIs, sentiment trends, and generate AI summaries
2. **Deep Dive Tab:** Explore raw data with advanced filtering
3. **About Tab:** Learn about the project architecture and features

## Data Architecture

The application generates 200 correlated reviews spanning 12 months. Negative reviews are programmatically injected with specific defect keywords:
- Battery/Performance issues (40%)
- App crashes/Bugs (30%)
- Customer support issues (30%)

This correlation ensures the AI has realistic patterns to analyze and identify root causes.

## License

MIT

