"""
News Engine Module
Fetches and analyzes news sentiment from GDELT API for XAU/USD
"""

import logging
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BULLISH_KEYWORDS, BEARISH_KEYWORDS,
    GDELT_BASE_URL, GDELT_QUERY, GDELT_MAX_RECORDS, GDELT_TIMEOUT,
    NEWS_CACHE_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class NewsEngine:
    """
    Fetches news from GDELT API and performs sentiment analysis
    using keyword-based NLP approach.
    """
    
    def __init__(self):
        logger.info("NewsEngine initialized")
        
        self.bullish_keywords = BULLISH_KEYWORDS
        self.bearish_keywords = BEARISH_KEYWORDS
        self.gdelt_url = GDELT_BASE_URL
        self.gdelt_query = GDELT_QUERY
        self.max_records = GDELT_MAX_RECORDS
        self.timeout = GDELT_TIMEOUT
        self.cache_file = NEWS_CACHE_FILE
    
    def fetch_gdelt_news(self) -> Optional[List[Dict]]:
        """
        Fetch gold-related news from GDELT API.
        
        Returns:
            List of news articles or None if request fails
        """
        logger.info(f"Fetching news from GDELT API (query: {self.gdelt_query})")
        
        params = {
            'query': self.gdelt_query,
            'mode': 'artlist',
            'maxrecords': self.max_records,
            'format': 'json',
            'lang': 'english'
        }
        
        try:
            response = requests.get(
                self.gdelt_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'articles' not in data or not data['articles']:
                logger.warning("No articles found in GDELT response")
                return []
            
            articles = data['articles']
            logger.info(f"Fetched {len(articles)} articles from GDELT")
            
            # Cache the results
            self._cache_news(articles)
            
            return articles
            
        except requests.exceptions.Timeout:
            logger.error(f"GDELT request timed out after {self.timeout} seconds")
            return self._load_cached_news()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"GDELT request failed: {str(e)}")
            return self._load_cached_news()
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GDELT JSON response: {str(e)}")
            return self._load_cached_news()
            
        except Exception as e:
            logger.error(f"Unexpected error fetching news: {str(e)}")
            return self._load_cached_news()
    
    def _cache_news(self, articles: List[Dict]) -> bool:
        """
        Cache news articles to disk.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            bool: Success status
        """
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'articles': articles
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Cached {len(articles)} articles to {self.cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching news: {str(e)}")
            return False
    
    def _load_cached_news(self) -> Optional[List[Dict]]:
        """
        Load cached news from disk.
        
        Returns:
            List of cached articles or None
        """
        if not os.path.exists(self.cache_file):
            logger.warning(f"No cached news found at {self.cache_file}")
            return []
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            articles = cache_data.get('articles', [])
            cache_time = cache_data.get('timestamp', 'unknown')
            
            logger.info(f"Loaded {len(articles)} cached articles from {cache_time}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error loading cached news: {str(e)}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of a text using keyword-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not text:
            return {
                'score': 0.0,
                'label': 'NEUTRAL',
                'bullish_count': 0,
                'bearish_count': 0
            }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count bullish and bearish keywords
        bullish_count = sum(1 for word in words if any(kw in word for kw in self.bullish_keywords))
        bearish_count = sum(1 for word in words if any(kw in word for kw in self.bearish_keywords))
        
        total_words = len(words)
        if total_words == 0:
            return {
                'score': 0.0,
                'label': 'NEUTRAL',
                'bullish_count': bullish_count,
                'bearish_count': bearish_count
            }
        
        # Calculate sentiment score (-100 to +100)
        score = ((bullish_count - bearish_count) / total_words) * 100
        
        # Clamp score to [-100, 100]
        score = max(-100, min(100, score))
        
        # Determine label
        if score > 5:
            label = 'BULLISH'
        elif score < -5:
            label = 'BEARISH'
        else:
            label = 'NEUTRAL'
        
        result = {
            'score': round(score, 2),
            'label': label,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'total_words': total_words
        }
        
        logger.debug(f"Sentiment analysis: {result}")
        
        return result
    
    def get_market_sentiment(self) -> Dict[str, any]:
        """
        Get aggregated market sentiment from all news articles.
        
        Returns:
            dict: Aggregated sentiment with top headlines
        """
        logger.info("Computing market sentiment...")
        
        # Fetch news
        articles = self.fetch_gdelt_news()
        
        if not articles:
            logger.warning("No articles available for sentiment analysis")
            return {
                'score': 0.0,
                'label': 'NEUTRAL',
                'confidence': 0.0,
                'articles_analyzed': 0,
                'top_headlines': [],
                'sentiment_distribution': {'BULLISH': 0, 'NEUTRAL': 0, 'BEARISH': 0}
            }
        
        # Analyze each article
        sentiments = []
        top_headlines = []
        distribution = {'BULLISH': 0, 'NEUTRAL': 0, 'BEARISH': 0}
        
        for article in articles[:self.max_records]:
            title = article.get('title', '')
            url = article.get('url', '')
            
            # Analyze title sentiment
            sentiment = self.analyze_sentiment(title)
            sentiments.append(sentiment['score'])
            distribution[sentiment['label']] += 1
            
            # Store headline info
            top_headlines.append({
                'title': title,
                'sentiment': sentiment['label'],
                'score': sentiment['score'],
                'url': url
            })
        
        # Calculate aggregate sentiment
        avg_score = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        # Normalize score to [0, 1] range for signal engine
        normalized_score = (avg_score + 100) / 200  # Maps [-100, 100] to [0, 1]
        
        # Determine overall label
        if avg_score > 5:
            overall_label = 'BULLISH'
        elif avg_score < -5:
            overall_label = 'BEARISH'
        else:
            overall_label = 'NEUTRAL'
        
        # Calculate confidence based on agreement
        max_category = max(distribution.values())
        confidence = max_category / len(articles) if articles else 0.0
        
        # Sort headlines by absolute score (most impactful first)
        top_headlines.sort(key=lambda x: abs(x['score']), reverse=True)
        top_5_headlines = top_headlines[:5]
        
        result = {
            'score': round(avg_score, 2),
            'normalized_score': round(normalized_score, 3),
            'label': overall_label,
            'confidence': round(confidence, 3),
            'articles_analyzed': len(articles),
            'top_headlines': top_5_headlines,
            'sentiment_distribution': distribution,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Market sentiment: {overall_label} (score={avg_score:.2f}, "
                   f"articles={len(articles)})")
        
        return result
    
    def print_sentiment_report(self, sentiment: Dict[str, any]) -> None:
        """
        Print formatted sentiment report to console.
        
        Args:
            sentiment: Sentiment dictionary from get_market_sentiment()
        """
        print("\n" + "=" * 50)
        print("📰 NEWS SENTIMENT ANALYSIS")
        print("=" * 50)
        
        label_emoji = {
            'BULLISH': '🐂',
            'BEARISH': '🐻',
            'NEUTRAL': '➖'
        }
        
        emoji = label_emoji.get(sentiment['label'], '❓')
        print(f"{emoji} Overall Sentiment: {sentiment['label']}")
        print(f"📊 Score: {sentiment['score']:.2f}")
        print(f"📈 Normalized: {sentiment['normalized_score']:.3f}")
        print(f"🎯 Confidence: {sentiment['confidence']*100:.1f}%")
        print(f"📄 Articles Analyzed: {sentiment['articles_analyzed']}")
        
        print("\nDistribution:")
        for label, count in sentiment['sentiment_distribution'].items():
            bar = '█' * int(count / max(sentiment['articles_analyzed'], 1) * 20)
            print(f"  {label}: {bar} ({count})")
        
        if sentiment['top_headlines']:
            print("\nTop Headlines:")
            for i, headline in enumerate(sentiment['top_headlines'][:5], 1):
                sent_emoji = label_emoji.get(headline['sentiment'], '')
                print(f"  {i}. {sent_emoji} {headline['title'][:60]}...")
        
        print("=" * 50)
