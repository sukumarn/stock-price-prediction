import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import yfinance as yf
from typing import List, Dict, Tuple
import logging

class NewsAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.company_name = self._get_company_name()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_company_name(self) -> str:
        """Get the company name from the stock symbol using yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            return ticker.info.get('longName', self.symbol)
        except Exception as e:
            self.logger.warning(f"Could not get company name for {self.symbol}: {e}")
            return self.symbol

    def get_news_articles(self) -> List[Dict]:
        """Fetch recent news articles about the company"""
        articles = []
        search_query = f"{self.company_name} stock market news"
        
        try:
            # Using Google News
            url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            
            items = soup.find_all('item')
            for item in items[:5]:  # Get top 5 news articles
                article = {
                    'title': item.title.text,
                    'link': item.link.text,
                    'date': datetime.strptime(item.pubDate.text, '%a, %d %b %Y %H:%M:%S %Z'),
                    'description': item.description.text
                }
                articles.append(article)
                
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            
        return articles

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze the sentiment of a text using TextBlob"""
        analysis = TextBlob(text)
        
        # Get polarity score (-1 to 1)
        polarity = analysis.sentiment.polarity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return polarity, sentiment

    def get_news_sentiment_analysis(self) -> pd.DataFrame:
        """Get news articles and their sentiment analysis"""
        articles = self.get_news_articles()
        results = []
        
        for article in articles:
            polarity, sentiment = self.analyze_sentiment(article['title'] + " " + article['description'])
            
            result = {
                'Date': article['date'],
                'Title': article['title'],
                'Description': article['description'],
                'Link': article['link'],
                'Sentiment': sentiment,
                'Sentiment_Score': polarity
            }
            results.append(result)
            
        df = pd.DataFrame(results)
        
        # Calculate aggregate sentiment metrics
        avg_sentiment = df['Sentiment_Score'].mean()
        sentiment_counts = df['Sentiment'].value_counts()
        
        self.logger.info(f"\nAggregate Sentiment Analysis for {self.company_name}:")
        self.logger.info(f"Average Sentiment Score: {avg_sentiment:.2f}")
        self.logger.info("\nSentiment Distribution:")
        self.logger.info(sentiment_counts)
        
        return df

    def save_sentiment_analysis(self, output_dir: str = 'output'):
        """Save sentiment analysis results to CSV"""
        df = self.get_news_sentiment_analysis()
        filename = f"{output_dir}/{self.symbol}_news_sentiment.csv"
        df.to_csv(filename, index=False)
        self.logger.info(f"\nSentiment analysis saved to {filename}")
        return df

    def get_trading_signal(self) -> str:
        """Generate trading signal based on news sentiment"""
        df = self.get_news_sentiment_analysis()
        avg_sentiment = df['Sentiment_Score'].mean()
        
        if avg_sentiment > 0.2:
            return "Strong Buy"
        elif avg_sentiment > 0.1:
            return "Buy"
        elif avg_sentiment < -0.2:
            return "Strong Sell"
        elif avg_sentiment < -0.1:
            return "Sell"
        else:
            return "Hold" 