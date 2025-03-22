import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import yfinance as yf
from typing import List, Dict, Tuple
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newspaper import Article
import numpy as np

class NewsAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.company_name = self._get_company_name()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()

    def _get_company_name(self) -> str:
        """Get the company name from the stock symbol using yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            return ticker.info.get('longName', self.symbol)
        except Exception as e:
            self.logger.warning(f"Could not get company name for {self.symbol}: {e}")
            return self.symbol

    def get_news_articles(self) -> List[Dict]:
        """Fetch recent news articles about the company from multiple sources"""
        articles = []
        search_query = f"{self.company_name} stock market news"
        
        # 1. Google News
        try:
            url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            
            items = soup.find_all('item')
            for item in items[:3]:  # Get top 3 articles from Google News
                article = {
                    'title': item.title.text,
                    'link': item.link.text,
                    'date': datetime.strptime(item.pubDate.text, '%a, %d %b %Y %H:%M:%S %Z'),
                    'description': item.description.text,
                    'source': 'Google News'
                }
                articles.append(article)
        except Exception as e:
            self.logger.error(f"Error fetching from Google News: {e}")

        # 2. Yahoo Finance
        try:
            ticker = yf.Ticker(self.symbol)
            news = ticker.news
            for item in news[:3]:  # Get top 3 articles from Yahoo Finance
                article = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'description': item.get('summary', ''),
                    'source': 'Yahoo Finance'
                }
                articles.append(article)
        except Exception as e:
            self.logger.error(f"Error fetching from Yahoo Finance: {e}")

        return articles

    def _get_detailed_content(self, url: str) -> str:
        """Extract detailed content from article URL"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            self.logger.warning(f"Could not extract detailed content from {url}: {e}")
            return ""

    def analyze_sentiment(self, text: str) -> Dict:
        """Enhanced sentiment analysis using multiple methods"""
        # 1. TextBlob Analysis
        blob_analysis = TextBlob(text)
        polarity = blob_analysis.sentiment.polarity
        subjectivity = blob_analysis.sentiment.subjectivity
        
        # 2. VADER Analysis
        vader_scores = self.sia.polarity_scores(text)
        
        # 3. Combine scores with weights
        combined_score = (polarity + vader_scores['compound']) / 2
        
        # 4. Calculate confidence and intensity
        confidence = abs(combined_score) * (1 - subjectivity)
        intensity = abs(combined_score) * vader_scores['neu']
        
        # 5. Determine sentiment category with fine-grained labels
        if combined_score > 0.3:
            sentiment = "Very Positive"
        elif combined_score > 0.1:
            sentiment = "Positive"
        elif combined_score < -0.3:
            sentiment = "Very Negative"
        elif combined_score < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            'sentiment': sentiment,
            'score': combined_score,
            'confidence': confidence,
            'intensity': intensity,
            'subjectivity': subjectivity,
            'vader_scores': vader_scores
        }

    def get_news_sentiment_analysis(self) -> pd.DataFrame:
        """Get enhanced news sentiment analysis"""
        articles = self.get_news_articles()
        results = []
        
        for article in articles:
            # Get detailed content when available
            detailed_content = self._get_detailed_content(article['link'])
            text_to_analyze = f"{article['title']} {article['description']} {detailed_content}"
            
            sentiment_results = self.analyze_sentiment(text_to_analyze)
            
            result = {
                'Date': article['date'],
                'Title': article['title'],
                'Description': article['description'],
                'Link': article['link'],
                'Source': article['source'],
                'Sentiment': sentiment_results['sentiment'],
                'Sentiment_Score': sentiment_results['score'],
                'Confidence': sentiment_results['confidence'],
                'Intensity': sentiment_results['intensity'],
                'Subjectivity': sentiment_results['subjectivity'],
                'VADER_Negative': sentiment_results['vader_scores']['neg'],
                'VADER_Neutral': sentiment_results['vader_scores']['neu'],
                'VADER_Positive': sentiment_results['vader_scores']['pos'],
                'VADER_Compound': sentiment_results['vader_scores']['compound']
            }
            results.append(result)
            
        df = pd.DataFrame(results)
        
        # Calculate advanced metrics
        self._calculate_advanced_metrics(df)
        
        return df

    def _calculate_advanced_metrics(self, df: pd.DataFrame):
        """Calculate and log advanced sentiment metrics"""
        metrics = {
            'Average Sentiment Score': df['Sentiment_Score'].mean(),
            'Sentiment Volatility': df['Sentiment_Score'].std(),
            'Average Confidence': df['Confidence'].mean(),
            'Average Intensity': df['Intensity'].mean(),
            'Average Subjectivity': df['Subjectivity'].mean(),
            'VADER Metrics': {
                'Negative': df['VADER_Negative'].mean(),
                'Neutral': df['VADER_Neutral'].mean(),
                'Positive': df['VADER_Positive'].mean(),
                'Compound': df['VADER_Compound'].mean()
            }
        }
        
        self.logger.info("\n=== Advanced Sentiment Metrics ===")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                self.logger.info(f"\n{metric}:")
                for sub_metric, sub_value in value.items():
                    self.logger.info(f"  {sub_metric}: {sub_value:.3f}")
            else:
                self.logger.info(f"{metric}: {value:.3f}")

    def get_trading_signal(self) -> Tuple[str, float]:
        """Generate detailed trading signal based on enhanced sentiment analysis"""
        df = self.get_news_sentiment_analysis()
        
        # Calculate weighted sentiment score
        weighted_scores = df.apply(lambda row: 
            row['Sentiment_Score'] * row['Confidence'] * (1 + row['Intensity']), axis=1)
        
        avg_weighted_sentiment = weighted_scores.mean()
        sentiment_volatility = df['Sentiment_Score'].std()
        
        # Generate signal with confidence level
        confidence_level = (1 - sentiment_volatility) * df['Confidence'].mean()
        
        if avg_weighted_sentiment > 0.3:
            signal = "Strong Buy"
        elif avg_weighted_sentiment > 0.1:
            signal = "Buy"
        elif avg_weighted_sentiment < -0.3:
            signal = "Strong Sell"
        elif avg_weighted_sentiment < -0.1:
            signal = "Sell"
        else:
            signal = "Hold"
            
        return signal, confidence_level 