"""
═══════════════════════════════════════════════════════════════════
NEWS ENGINE - GDELT + NLP SENTIMENT ANALYSIS
═══════════════════════════════════════════════════════════════════
Real-time news sentiment analysis for gold market using GDELT API
and keyword-based NLP with caching and fallback mechanisms.
═══════════════════════════════════════════════════════════════════
"""

import logging
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

from config import (
    GDELT_URL, GDELT_QUERY, NEWS_CACHE_MINUTES, MAX_NEWS_ARTICLES,
    BULLISH_GOLD_KEYWORDS, BEARISH_GOLD_KEYWORDS, NEWS_CACHE_PATH
)

logger = logging.getLogger(__name__)


class NewsEngine:
    """
    Institutional news sentiment engine using GDELT API and NLP.
    Implements caching, fallback, and gold-specific sentiment analysis.
    """
    
    def __init__(self, cache_path: Path = NEWS_CACHE_PATH):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.NewsEngine")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "XAUUSD-Institutional-Analyzer/1.0"
        })
    
    def fetch_gdelt_news(
        self, 
        max_records: int = 30,
        timeout: int = 12
    ) -> List[Dict]:
        """
        Fetch gold-related news from GDELT API.
        
        Args:
            max_records: Maximum articles to fetch
            timeout: Request timeout in seconds
            
        Returns:
            List of article dicts with title, url, date, source
        """
        self.logger.info(f"Fetching news from GDELT (max {max_records} articles)...")
        
        params = {
            "query": GDELT_QUERY,
            "mode": "artlist",
            "maxrecords": str(max_records),
            "format": "json",
            "timespan": "24h"
        }
        
        try:
            response = self.session.get(
                GDELT_URL,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "articles" not in data:
                self.logger.warning("No articles in GDELT response")
                return self._load_from_cache()
            
            articles = []
            for art in data["articles"][:max_records]:
                article = {
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "date": art.get("seendate", ""),
                    "source": art.get("domain", "unknown"),
                    "snippet": art.get("snippet", "")
                }
                articles.append(article)
            
            self.logger.info(f"Fetched {len(articles)} articles from GDELT")
            
            # Cache successful fetch
            self._save_to_cache(articles)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"GDELT request failed: {e}")
            return self._load_from_cache()
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse GDELT response: {e}")
            return self._load_from_cache()
        except Exception as e:
            self.logger.error(f"Unexpected error fetching news: {e}")
            return self._load_from_cache()
    
    def _save_to_cache(self, articles: List[Dict]):
        """Save articles to cache with timestamp."""
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "articles": articles
        }
        
        try:
            with open(self.cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
            self.logger.debug(f"Saved {len(articles)} articles to cache")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def _load_from_cache(self) -> List[Dict]:
        """Load articles from cache if available and fresh."""
        if not self.cache_path.exists():
            self.logger.warning("No cache file found")
            return []
        
        try:
            with open(self.cache_path, "r") as f:
                cache_data = json.load(f)
            
            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            
            if age_minutes > NEWS_CACHE_MINUTES * 2:  # Allow 2x age
                self.logger.info(f"Cache too old ({age_minutes:.0f} min)")
                return []
            
            articles = cache_data.get("articles", [])
            self.logger.info(f"Loaded {len(articles)} articles from cache")
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
            return []
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of text using keyword matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with score (-1 to 1), label, and hit counts
        """
        text_lower = text.lower()
        
        bull_hits = sum(1 for kw in BULLISH_GOLD_KEYWORDS if kw in text_lower)
        bear_hits = sum(1 for kw in BEARISH_GOLD_KEYWORDS if kw in text_lower)
        
        total_words = max(len(text.split()), 1)
        total_hits = bull_hits + bear_hits
        
        # Raw score: -1 (bearish) to 1 (bullish)
        if total_hits == 0:
            raw_score = 0.0
        else:
            raw_score = (bull_hits - bear_hits) / (total_hits + 1)
        
        # Apply gold safe-haven multiplier
        crisis_keywords = ["crisis", "war", "conflict", "panic", "crash", "fear"]
        if any(kw in text_lower for kw in crisis_keywords) and bull_hits > bear_hits:
            raw_score *= 1.3  # Amplify bullish during crisis
        
        # Clip to [-1, 1]
        raw_score = max(-1.0, min(1.0, raw_score))
        
        # Determine label
        if raw_score > 0.15:
            label = "BULLISH"
        elif raw_score < -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"
        
        return {
            "score": raw_score,
            "label": label,
            "bull_hits": bull_hits,
            "bear_hits": bear_hits,
            "total_hits": total_hits
        }
    
    def get_market_sentiment(self) -> Dict:
        """
        Aggregate sentiment from all recent news articles.
        
        Returns:
            Dict with aggregated sentiment metrics
        """
        self.logger.info("Computing market sentiment...")
        
        articles = self.fetch_gdelt_news(MAX_NEWS_ARTICLES)
        
        if not articles:
            self.logger.warning("No articles available for sentiment analysis")
            return {
                "score": 0.0,
                "norm": 0.5,
                "label": "NEUTRAL",
                "article_count": 0,
                "top_headlines": [],
                "source": "none"
            }
        
        scores = []
        analyzed_articles = []
        
        for article in articles:
            # Analyze title + snippet
            text = f"{article['title']} {article['snippet']}"
            result = self.analyze_text(text)
            
            scores.append(result["score"])
            analyzed_articles.append({
                **article,
                "sentiment": result
            })
        
        # Aggregate scores
        mean_score = np.mean(scores) if scores else 0.0
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        
        # Normalize to [0, 1]
        norm_score = (mean_score + 1) / 2
        
        # Determine overall label
        if norm_score > 0.58:
            label = "BULLISH"
        elif norm_score < 0.42:
            label = "BEARISH"
        else:
            label = "NEUTRAL"
        
        # Get top headlines by sentiment extremity
        sorted_articles = sorted(
            analyzed_articles,
            key=lambda x: abs(x["sentiment"]["score"]),
            reverse=True
        )
        top_headlines = [a["title"] for a in sorted_articles[:5]]
        
        self.logger.info(
            f"Sentiment: {label} (score={mean_score:.3f}, norm={norm_score:.3f})"
        )
        
        return {
            "score": mean_score,
            "norm": norm_score,
            "label": label,
            "article_count": len(articles),
            "top_headlines": top_headlines,
            "source": "live",
            "std": std_score,
            "analyzed_articles": analyzed_articles[:10]  # Top 10 for reference
        }


# Import numpy at module level for use in methods
import numpy as np
