"""
Data Generation Utilities for VOC Insight Engine
Generates correlated synthetic review data with realistic defect patterns.
"""

import pandas as pd
import numpy as np
from faker import Faker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import random

# Initialize VADER (Faker will be created per call with seed)
analyzer = SentimentIntensityAnalyzer()


def generate_correlated_reviews(n_reviews=200, months_back=12, seed=None):
    """
    Generate correlated review data where negative reviews contain specific defect keywords.
    
    Args:
        n_reviews: Number of reviews to generate
        months_back: Number of months to span back from today
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: date, product, review_text, sentiment_score, sentiment_category
    """
    # Set seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create Faker instance with seed if provided
    fake = Faker()
    if seed is not None:
        Faker.seed(seed)
    
    reviews = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    # Define defect scenarios with their keywords
    defect_scenarios = {
        'battery': {
            'weight': 0.4,
            'keywords': ['battery drain', 'overheating', 'charge', 'battery life', 'power consumption']
        },
        'crash': {
            'weight': 0.3,
            'keywords': ['crash', 'bug', 'app close', 'freeze', 'not responding', 'error']
        },
        'support': {
            'weight': 0.3,
            'keywords': ['customer support', 'rude', 'wait time', 'unhelpful', 'slow response', 'poor service']
        }
    }
    
    # Generate reviews
    for i in range(n_reviews):
        # Random date within the range
        random_days = random.randint(0, months_back * 30)
        review_date = start_date + timedelta(days=random_days)
        
        # Random product category
        product = random.choice(['Mobile App', 'Web Platform', 'Desktop Software', 'API Service'])
        
        # Determine if this will be a negative review (30% chance)
        is_negative = random.random() < 0.3
        
        if is_negative:
            # Select defect scenario based on weights
            scenario_choice = random.random()
            cumulative = 0
            selected_scenario = None
            
            for scenario_name, scenario_data in defect_scenarios.items():
                cumulative += scenario_data['weight']
                if scenario_choice <= cumulative:
                    selected_scenario = scenario_data
                    break
            
            # Generate negative review with defect keywords
            base_complaints = [
                f"I'm experiencing {selected_scenario['keywords'][0]} issues.",
                f"The {selected_scenario['keywords'][0]} is really frustrating.",
                f"Having problems with {selected_scenario['keywords'][0]} lately.",
                f"Noticed {selected_scenario['keywords'][0]} after the update.",
            ]
            
            # Add 1-2 more keywords from the scenario
            additional_keywords = random.sample(
                selected_scenario['keywords'][1:], 
                min(2, len(selected_scenario['keywords']) - 1)
            )
            
            review_text = random.choice(base_complaints)
            review_text += f" Also dealing with {', '.join(additional_keywords)}. "
            review_text += fake.text(max_nb_chars=100)
            
        else:
            # Generate positive/neutral review
            positive_templates = [
                "Great product! Love it.",
                "Fast and reliable. Highly recommend.",
                "Excellent service and quality.",
                "Works perfectly for my needs.",
                "Very satisfied with the experience.",
                "Outstanding performance and features.",
                "Best in class. No complaints.",
            ]
            review_text = random.choice(positive_templates)
            review_text += " " + fake.text(max_nb_chars=80)
        
        # Calculate sentiment using VADER
        sentiment_scores = analyzer.polarity_scores(review_text)
        sentiment_score = sentiment_scores['compound']
        
        # Categorize sentiment
        if sentiment_score >= 0.05:
            sentiment_category = 'Positive'
        elif sentiment_score <= -0.05:
            sentiment_category = 'Negative'
        else:
            sentiment_category = 'Neutral'
        
        reviews.append({
            'date': review_date,
            'product': product,
            'review_text': review_text,
            'sentiment_score': sentiment_score,
            'sentiment_category': sentiment_category
        })
    
    df = pd.DataFrame(reviews)
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def calculate_nps(df):
    """
    Calculate Net Promoter Score from sentiment data.
    NPS = % Promoters (Positive) - % Detractors (Negative)
    """
    total = len(df)
    if total == 0:
        return 0
    
    promoters = len(df[df['sentiment_category'] == 'Positive'])
    detractors = len(df[df['sentiment_category'] == 'Negative'])
    
    nps = ((promoters - detractors) / total) * 100
    return round(nps, 1)

