import streamlit as st
import pandas as pd
import ossource venv/bin/activate
import time
import random
from datetime import datetime, timedelta
import re
import feedparser
import requests
from bs4 import BeautifulSoup
import logging
import html
import plotly.express as px
import plotly.graph_objects as go
import json
import warnings
from anthropic import Anthropic
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Initialize NLTK data immediately
def ensure_nltk_resources():
    """Make sure required NLTK resources are available"""
    import nltk
    import os
    
    # Create a directory for NLTK data if it doesn't exist
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # List of required resources
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    # Download resources if not already available
    for resource, path in required_resources:
        try:
            nltk.data.find(path)
            print(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)

# Call this function to ensure NLTK resources are available
ensure_nltk_resources()

# Try to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is not installed, continue without it
    pass

# Suppress the ScriptRunContext warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

class RSSNewsScraperMultiSource:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com/'
        }
        self.all_articles = []
        
        # Create output directory if it doesn't exist
        self.output_dir = 'news_data'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # RSS feeds by category
        self.rss_feeds = {
            'US Politics': [
                {
                    'url': 'https://www.cnbc.com/id/10000113/device/rss/rss.html',
                    'source': 'CNBC Politics'
                },
                {
                    'url': 'http://feeds.washingtonpost.com/rss/politics',
                    'source': 'Washington Post Politics'
                },
                {
                    'url': 'https://feeds.npr.org/1014/rss.xml',
                    'source': 'NPR Politics'
                },
                {
                    'url': 'https://www.politico.com/rss/politicopicks.xml',
                    'source': 'Politico'
                },
                {
                    'url': 'https://thehill.com/rss/syndicator/19109',
                    'source': 'The Hill'
                }
            ],
            'Brazil Politics': [
                {
                    'url': 'https://feeds.folha.uol.com.br/poder/rss091.xml',
                    'source': 'Folha - Poder'
                },
                {
                    'url': 'https://g1.globo.com/rss/g1/politica/',
                    'source': 'G1 Política'
                },
                {
                    'url': 'https://www.poder360.com.br/feed/',
                    'source': 'Poder360'
                }
            ],
            'LATAM Politics': [
                {
                    'url': 'https://news.google.com/rss/search?q=latin+america+politics+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - LATAM Politics'
                },
                {
                    'url': 'https://www.reuters.com/arc/outboundfeeds/v3/category/latin-america/?outputType=xml',
                    'source': 'Reuters LATAM'
                }
            ],
            'Global Politics': [
                {
                    'url': 'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
                    'source': 'WSJ World'
                },
                {
                    'url': 'http://feeds.bbci.co.uk/news/world/rss.xml',
                    'source': 'BBC World'
                },
                {
                    'url': 'https://news.un.org/feed/subscribe/en/news/all/rss.xml',
                    'source': 'UN News'
                }
            ],
            'US Finance': [
                {
                    'url': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
                    'source': 'CNBC Finance'
                },
                {
                    'url': 'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml',
                    'source': 'Wall Street Journal'
                },
                {
                    'url': 'http://feeds.marketwatch.com/marketwatch/topstories/',
                    'source': 'MarketWatch'
                },
                {
                    'url': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258',
                    'source': 'CNBC Markets'
                }
            ],
            'Europe Finance': [
                {
                    'url': 'https://www.ft.com/rss/home',
                    'source': 'Financial Times'
                },
                {
                    'url': 'http://feeds.bbci.co.uk/news/business/rss.xml',
                    'source': 'BBC Business'
                },
                {
                    'url': 'https://news.google.com/rss/search?q=europe+finance+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - Europe Finance'
                }
            ],
            'LATAM Finance': [
                {
                    'url': 'https://news.google.com/rss/search?q=latin+america+finance+economy+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - LATAM Finance'
                }
            ],
            'Brazil Finance': [
                {
                    'url': 'https://www.infomoney.com.br/feed/',
                    'source': 'InfoMoney'
                },
                {
                    'url': 'https://g1.globo.com/rss/g1/economia/',
                    'source': 'G1 Economia'
                },
                {
                    'url': 'https://agenciabrasil.ebc.com.br/rss/ultimasnoticias/feed.xml',
                    'source': 'Agência Brasil'
                }
            ],
            'Global Finance': [
                {
                    'url': 'https://news.google.com/rss/search?q=global+financial+markets+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - Global Markets'
                },
                {
                    'url': 'https://www.imf.org/en/News/Rss',
                    'source': 'IMF'
                },
                {
                    'url': 'https://www.bloomberg.com/feed/markets/sitemap_index.xml',
                    'source': 'Bloomberg Markets'
                }
            ],
            'China Finance': [
                {
                    'url': 'https://news.google.com/rss/search?q=china+economy+finance+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - China Finance'
                },
                {
                    'url': 'https://www.scmp.com/rss/4/feed',
                    'source': 'South China Morning Post - Economy'
                }
            ],
            'Canada Finance': [
                {
                    'url': 'https://news.google.com/rss/search?q=canada+economy+finance+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - Canada Finance'
                }
            ],
            'Business': [
                {
                    'url': 'https://www.cnbc.com/id/10001147/device/rss/rss.html',
                    'source': 'CNBC Business'
                },
                {
                    'url': 'https://www.ft.com/companies/rss',
                    'source': 'Financial Times - Companies'
                }
            ],
            'M&A': [
                {
                    'url': 'https://news.google.com/rss/search?q=merger+acquisition+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - M&A'
                },
                {
                    'url': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100345817',
                    'source': 'CNBC Deals and IPOs'
                }
            ],
            'Macroeconomics': [
                {
                    'url': 'https://news.google.com/rss/search?q=macroeconomics+inflation+rates+gdp+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - Macroeconomics'
                }
            ],
            'Microeconomics': [
                {
                    'url': 'https://news.google.com/rss/search?q=microeconomics+consumer+behavior+market+structure+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - Microeconomics'
                }
            ],
            'Trade War': [
                {
                    'url': 'https://news.google.com/rss/search?q=trade+war+tariffs+when:2d&hl=en-US&gl=US&ceid=US:en',
                    'source': 'Google News - Trade War'
                }
            ]
        }
    
    def setup_logging(self):
        """Set up basic logging to a file"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'rss_scraper_log.txt')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('RSSNewsScraperLogger')
        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        self.logger.info("RSS Scraper initialized")
    
    def log(self, message, level='info'):
        """Wrapper for logging with fallback to print"""
        try:
            if level == 'info':
                self.logger.info(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'warning':
                self.logger.warning(message)
        except:
            # Fallback to print if logging fails
            print(f"{level.upper()}: {message}")
    
    def clean_text(self, text):
        """Clean up text by removing HTML tags, decoding HTML entities, and normalizing whitespace"""
        if text is None:
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)
        
        return text.strip()
    
    def create_simple_summary(self, text, max_length=200):
        """Create a simple summary by truncating text to specified length"""
        if not text:
            return ""
        
        clean_text = self.clean_text(text)
        
        # If text is already short, return it as is
        if len(clean_text) <= max_length:
            return clean_text
        
        # Find the last period, question mark, or exclamation point within the max_length
        last_sentence_end = max(
            clean_text[:max_length].rfind('.'),
            clean_text[:max_length].rfind('?'),
            clean_text[:max_length].rfind('!')
        )
        
        # If found a sentence end, return up to that point
        if last_sentence_end > 0:
            return clean_text[:last_sentence_end + 1]
        
        # If no sentence end found, find the last space before max_length
        last_space = clean_text[:max_length].rfind(' ')
        if last_space > 0:
            return clean_text[:last_space] + '...'
        
        # If no space found, just truncate and add ellipsis
        return clean_text[:max_length] + '...'
    
    def is_recent_entry(self, entry, days=2):
        """Check if an entry is within the specified number of days"""
        # Try different date fields
        pub_date = None
        if hasattr(entry, 'published'):
            pub_date = entry.published
        elif hasattr(entry, 'pubDate'):
            pub_date = entry.pubDate
        elif hasattr(entry, 'updated'):
            pub_date = entry.updated
        
        if not pub_date:
            # If no date found, assume it's recent
            return True
        
        try:
            # Try to parse the date
            date_obj = None
            
            # Try different date formats
            formats_to_try = [
                '%a, %d %b %Y %H:%M:%S %z',  # RFC 822
                '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822 with timezone name
                '%Y-%m-%dT%H:%M:%S%z',      # ISO 8601
                '%Y-%m-%dT%H:%M:%SZ',       # ISO 8601 UTC
                '%Y-%m-%d %H:%M:%S',        # Basic format
                '%a %b %d %H:%M:%S %z %Y'   # Twitter format
            ]
            
            for fmt in formats_to_try:
                try:
                    date_obj = datetime.strptime(pub_date, fmt)
                    break
                except:
                    continue
            
            # If no format worked, try email utils
            if not date_obj:
                try:
                    from email.utils import parsedate_to_datetime
                    date_obj = parsedate_to_datetime(pub_date)
                except:
                    # If all parsing fails, return True (assume it's recent)
                    return True
            
            # Check if the entry is within the specified number of days
            cutoff_date = datetime.now() - timedelta(days=days)
            return date_obj >= cutoff_date
            
        except Exception as e:
            self.log(f"Error parsing date: {pub_date} - {str(e)}", 'warning')
            # If there's an error, return True (assume it's recent)
            return True
    
    def get_feed_data(self, feed_url, source_name, category):
        """Parse RSS feed and extract article information"""
        self.log(f"Fetching RSS feed: {feed_url} for {source_name}")
        
        try:
            # Add a small random delay to avoid too many simultaneous requests
            time.sleep(random.uniform(0.5, 2))
            
            # Parse the feed
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                self.log(f"No entries found in feed for {source_name}", 'warning')
                return []
            
            self.log(f"Found {len(feed.entries)} entries in feed for {source_name}")
            
            # Process entries (only those from the past 2 days)
            articles = []
            for entry in feed.entries:
                try:
                    # Skip if not recent
                    if not self.is_recent_entry(entry, days=2):
                        continue
                    
                    # Extract data from entry
                    title = entry.title if hasattr(entry, 'title') else "No title"
                    link = entry.link if hasattr(entry, 'link') else ""
                    
                    # Try different fields for summary/description
                    summary = ""
                    if hasattr(entry, 'summary'):
                        summary = entry.summary
                    elif hasattr(entry, 'description'):
                        summary = entry.description
                    elif hasattr(entry, 'content'):
                        # Some feeds use content instead of summary
                        summary = entry.content[0].value if entry.content else ""
                    
                    # Clean the summary text
                    summary = self.clean_text(summary)
                    
                    # Get publication date
                    pub_date = None
                    if hasattr(entry, 'published'):
                        pub_date = entry.published
                    elif hasattr(entry, 'pubDate'):
                        pub_date = entry.pubDate
                    elif hasattr(entry, 'updated'):
                        pub_date = entry.updated
                    
                    # Format date or use current date
                    if pub_date:
                        try:
                            # Try to parse the date, but use current date as fallback
                            date_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                            pub_date = date_obj.strftime("%Y-%m-%d")
                        except:
                            try:
                                # Try alternative format
                                from email.utils import parsedate_to_datetime
                                date_obj = parsedate_to_datetime(pub_date)
                                pub_date = date_obj.strftime("%Y-%m-%d")
                            except:
                                # If parsing fails, use the original string
                                pass
                    else:
                        pub_date = datetime.now().strftime("%Y-%m-%d")
                    
                    # Add to articles list
                    articles.append({
                        'headline': self.clean_text(title),
                        'summary': summary,
                        'url': link,
                        'source': source_name,
                        'category': category,
                        'timestamp': pub_date
                    })
                    
                except Exception as e:
                    self.log(f"Error processing entry for {source_name}: {str(e)}", 'error')
            
            self.log(f"Successfully processed {len(articles)} articles from {source_name}")
            return articles
            
        except Exception as e:
            self.log(f"Error fetching feed {feed_url} for {source_name}: {str(e)}", 'error')
            return []
    
    def scrape_category(self, category):
        """Scrape all feeds for a specific category"""
        self.log(f"Scraping category: {category}")
        
        if category not in self.rss_feeds:
            self.log(f"Unknown category: {category}", 'error')
            return
        
        for feed in self.rss_feeds[category]:
            try:
                feed_url = feed['url']
                source_name = feed['source']
                
                # Get articles from this feed
                articles = self.get_feed_data(feed_url, source_name, category)
                
                # Add to master list
                self.all_articles.extend(articles)
                
            except Exception as e:
                self.log(f"Error processing feed {feed} for category {category}: {str(e)}", 'error')
    
    def scrape_all_categories(self):
        """Scrape all categories defined in rss_feeds with no article limit"""
        self.log(f"Starting to scrape all categories for past 2 days")
        
        for category in self.rss_feeds.keys():
            try:
                self.scrape_category(category)
            except Exception as e:
                self.log(f"Error scraping category {category}: {str(e)}", 'error')
        
        self.log(f"Completed scraping all categories. Collected {len(self.all_articles)} articles total.")
    
    def save_results(self):
        """Save scraped articles to CSV files with error handling"""
        if not self.all_articles:
            self.log("No articles to save.", 'warning')
            return
        
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save all articles to one file
            df_all = pd.DataFrame(self.all_articles)
            all_file = os.path.join(self.output_dir, f"all_news_{timestamp}.csv")
            df_all.to_csv(all_file, index=False, encoding='utf-8-sig')
            self.log(f"Saved all {len(self.all_articles)} articles to {all_file}")
            
            # Save separate files by category
            categories = df_all['category'].unique()
            for category in categories:
                try:
                    df_category = df_all[df_all['category'] == category]
                    category_file = os.path.join(self.output_dir, f"{category.replace(' ', '_').lower()}_{timestamp}.csv")
                    df_category.to_csv(category_file, index=False, encoding='utf-8-sig')
                    self.log(f"Saved {len(df_category)} {category} articles to {category_file}")
                except Exception as e:
                    self.log(f"Error saving category {category}: {str(e)}", 'error')
        
        except Exception as e:
            self.log(f"Error saving results: {str(e)}", 'error')
            # Try a simplified approach as fallback
            try:
                simple_file = os.path.join(self.output_dir, "news_backup.csv")
                pd.DataFrame(self.all_articles).to_csv(simple_file, index=False)
                self.log(f"Saved backup file to {simple_file}")
            except:
                self.log("Critical failure: Could not save any results", 'error')
    
    def remove_duplicates(self):
        """Remove duplicate articles based on URL and headline"""
        if not self.all_articles:
            return
        
        self.log(f"Removing duplicates from {len(self.all_articles)} articles")
        
        # Convert to DataFrame for easier deduplication
        df = pd.DataFrame(self.all_articles)
        
        # Drop duplicates based on URL (most reliable method)
        df_no_url_dupes = df.drop_duplicates(subset=['url'])
        
        # Also check for duplicates in headlines (different URLs might have same content)
        df_no_dupes = df_no_url_dupes.drop_duplicates(subset=['headline'])
        
        # Convert back to list of dictionaries
        self.all_articles = df_no_dupes.to_dict('records')
        
        self.log(f"After removing duplicates: {len(self.all_articles)} articles remain")


# UTILITY FUNCTIONS FOR TEXT ANALYSIS AND CLAUDE API

# Download NLTK resources when needed - UPDATED
def download_nltk_resources():
    """Download NLTK resources if they're not already available"""
    import os
    import nltk
    
    # Create a directory for NLTK data if it doesn't exist
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # List of required resources
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    # Download resources if not already available
    for resource, path in required_resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

# Extract keywords from text
def extract_keywords(texts, min_word_length=3, max_keywords=50):
    """Extract common keywords from a list of texts"""
    download_nltk_resources()
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add some domain-specific stopwords
    additional_stopwords = {'said', 'says', 'reported', 'according', 'reuters', 'news', 'new', 'year', 'years',
                          'may', 'day', 'week', 'month', 'time', 'people', 'percent', 'today', 'yesterday'}
    stop_words.update(additional_stopwords)
    
    # Combine all texts
    all_text = " ".join(texts).lower()
    
    # Tokenize
    words = word_tokenize(all_text)
    
    # Filter words
    filtered_words = [word for word in words 
                    if word.isalpha() and len(word) >= min_word_length 
                    and word not in stop_words]
    
    # Count frequency
    word_counts = Counter(filtered_words)
    
    # Get the most common words
    return word_counts.most_common(max_keywords)

# Function to create a word cloud
def generate_wordcloud(text_series, max_words=100):
    """Generate a word cloud from a series of texts"""
    download_nltk_resources()
    
    # Combine texts
    combined_text = " ".join(text_series.fillna(""))
    
    # Create word cloud
    stop_words = set(stopwords.words('english'))
    additional_stopwords = {'said', 'says', 'reported', 'according', 'reuters', 'news'}
    stop_words.update(additional_stopwords)
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        stopwords=stop_words,
        max_words=max_words,
        collocations=False
    ).generate(combined_text)
    
    return wordcloud

# Function to call Claude API for summary
def get_claude_summary(articles, api_key, max_articles_per_category=15):
    """Get a summary of articles by category using Claude API"""
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Group articles by category
    category_articles = {}
    for article in articles:
        category = article['category']
        if category not in category_articles:
            category_articles[category] = []
        category_articles[category].append(article)
    
    # Prepare summaries dictionary
    summaries = {}
    
    # Process each category
    for category, category_articles_list in category_articles.items():
        # Limit to max articles to avoid overwhelming Claude
        selected_articles = sorted(category_articles_list, 
                                key=lambda x: x.get('timestamp', ''), 
                                reverse=True)[:max_articles_per_category]
        
        # Format articles for Claude prompt
        articles_text = ""
        for i, article in enumerate(selected_articles, 1):
            articles_text += f"{i}. HEADLINE: {article.get('headline', 'No headline')}\n"
            articles_text += f"   SOURCE: {article.get('source', 'Unknown')}\n"
            articles_text += f"   DATE: {article.get('timestamp', 'Unknown')}\n"
            articles_text += f"   SUMMARY: {article.get('summary', 'No summary')[:250]}...\n\n"
        
        # Create prompt for Claude
        prompt = f"""
        You are an expert news analyst and summarizer. Below are recent news articles about {category}.
        
        Please analyze these articles and provide:
        1. A concise overall summary of key developments (3-5 sentences)
        2. The most important 2-3 themes or trends
        3. Any significant events, facts, or announcements
        4. A brief outlook (what to watch for next)
        
        Articles:
        {articles_text}
        
        Format your response as markdown with clear headers and bullet points where appropriate.
        """
        
        try:
            # Call Claude API
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the summary
            summary = response.content[0].text
            summaries[category] = summary
            
        except Exception as e:
            summaries[category] = f"Error generating summary: {str(e)}"
    
    return summaries

# Function to get articles from the previous day
def get_previous_day_articles(df, days_back=1):
    """Get articles from N days ago"""
    # Ensure timestamp is a datetime
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    target_date = datetime.now().date() - timedelta(days=days_back)
    return df[df['date'] == target_date]


# VISUALIZATION FUNCTIONS

def create_category_chart(df):
    """Create a bar chart of article counts by category"""
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    fig = px.bar(
        category_counts, 
        x='Category', 
        y='Count',
        title='Articles by Category',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        xaxis_title="",
        height=500
    )
    
    return fig

def create_source_chart(df):
    """Create a bar chart of article counts by source"""
    source_counts = df['source'].value_counts().head(15).reset_index()
    source_counts.columns = ['Source', 'Count']
    
    fig = px.bar(
        source_counts, 
        x='Count', 
        y='Source',
        title='Top 15 News Sources',
        orientation='h',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=500
    )
    
    return fig

def create_timeline_chart(df):
    """Create a timeline chart of article counts by date"""
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    date_counts = df.groupby('date').size().reset_index()
    date_counts.columns = ['Date', 'Count']
    
    fig = px.line(
        date_counts, 
        x='Date', 
        y='Count',
        title='Article Volume Over Time',
        markers=True
    )
    
    fig.update_layout(height=500)
    return fig

def create_keyword_chart(df, column='headline', top_n=20):
    """Create a bar chart of most common keywords"""
    # Combine text from specified column
    texts = df[column].dropna().tolist()
    
    # Extract keywords
    keywords = extract_keywords(texts, max_keywords=top_n)
    
    # Create dataframe for plotting
    keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Count'])
    
    fig = px.bar(
        keyword_df,
        x='Keyword',
        y='Count',
        title=f'Top {top_n} Keywords in {column.capitalize()}',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        xaxis_title="",
        height=500
    )
    
    return fig

def create_category_source_heatmap(df):
    """Create a heatmap of article counts by category and source"""
    # Count articles by category and source
    category_source_counts = df.groupby(['category', 'source']).size().reset_index()
    category_source_counts.columns = ['Category', 'Source', 'Count']
    
    # Pivot to create a matrix suitable for heatmap
    pivot_df = category_source_counts.pivot_table(
        values='Count', 
        index='Category', 
        columns='Source', 
        fill_value=0
    )
    
    # Keep only the top sources to avoid too many columns
    top_sources = df['source'].value_counts().head(10).index.tolist()
    pivot_df = pivot_df[pivot_df.columns.intersection(top_sources)]
    
    # Create the heatmap
    fig = px.imshow(
        pivot_df,
        title='Articles by Category and Top Sources',
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    
    fig.update_layout(height=600)
    return fig

def create_wordcloud_figure(df, column='headline'):
    """Create a matplotlib figure with a wordcloud"""
    wordcloud = generate_wordcloud(df[column].dropna())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud from {column.capitalize()}')
    
    return fig


# Function to generate sample data if scraper isn't available
def generate_sample_data():
    """Create sample news data for testing when scraper is unavailable"""
    sources = [
        'CNBC Finance', 'Wall Street Journal', 'Financial Times', 'BBC Business',
        'Politico', 'The Hill', 'InfoMoney', 'Folha - Poder', 'Google News - M&A'
    ]
    
    categories = [
        'US Finance', 'Global Finance', 'US Politics', 'Brazilian Politics', 
        'LATAM Politics', 'M&A', 'Macroeconomics', 'Trade War'
    ]
    
    headlines = [
        "Central Bank Raises Interest Rates Amid Inflation Concerns",
        "Government Announces New Infrastructure Package",
        "Tech Giants Face Antitrust Scrutiny",
        "Global Markets React to Trade Tensions",
        "Oil Prices Surge on Supply Concerns",
        "Election Results Impact Market Sentiment",
        "New Regulations for Banking Sector Announced",
        "Retail Sales Data Shows Economic Recovery",
        "Major Merger Announced Between Tech Companies",
        "Brazilian Currency Gains Against Dollar"
    ]
    
    summaries = [
        "The decision comes as inflation reached its highest level in a decade, prompting monetary policy tightening.",
        "The $2 trillion package aims to rebuild aging infrastructure and create millions of jobs over the next decade.",
        "Regulators are investigating potential anticompetitive practices among major technology companies.",
        "Global markets experienced volatility as trade negotiations between major economies stalled.",
        "Crude oil prices increased by 5% following production cuts and geopolitical tensions.",
        "Markets responded positively to the election outcome, with banking and healthcare sectors seeing gains.",
        "New regulatory framework aims to increase capital requirements and enhance consumer protections.",
        "Consumer spending rose 2.4% month-over-month, indicating economic resilience despite challenges.",
        "The merger will create a new entity valued at over $50 billion, pending regulatory approval.",
        "The Brazilian real strengthened against the U.S. dollar following positive economic data."
    ]
    
    data = []
    
    # Create sample articles
    for i in range(100):
        source_index = i % len(sources)
        source = sources[source_index]
        
        # Assign category based on source
        if source in ['CNBC Finance', 'Wall Street Journal']:
            category = 'US Finance'
        elif source in ['Financial Times', 'BBC Business']:
            category = 'Global Finance'
        elif source in ['Politico', 'The Hill']:
            category = 'US Politics'
        elif source in ['InfoMoney']:
            category = 'Brazil Finance'
        elif source in ['Folha - Poder']:
            category = 'Brazilian Politics'
        elif source in ['Google News - M&A']:
            category = 'M&A'
        else:
            category = categories[i % len(categories)]
        
        # Create a date within the past two weeks
        days_ago = i % 14
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Select headline and summary
        headline_index = i % len(headlines)
        headline = f"{headlines[headline_index]} - {i+1}"
        summary = summaries[headline_index]
        
        data.append({
            'headline': headline,
            'summary': summary,
            'url': f"https://example.com/{source.lower().replace(' ', '')}/article{i+1}",
            'source': source,
            'category': category,
            'timestamp': date
        })
    
    return data

# Function to get a list of saved news files
def get_saved_news_files():
    """Get a list of all saved news CSV files"""
    if not os.path.exists('news_data'):
        return []
    
    files = [f for f in os.listdir('news_data') if f.startswith("all_news_") and f.endswith(".csv")]
    return sorted(files, reverse=True)

# Function to load a specific news file
def load_news_file(filename):
    """Load a specific news CSV file"""
    filepath = os.path.join('news_data', filename)
    try:
        df = pd.read_csv(filepath)
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading file {filename}: {str(e)}")
        return []

# Set up the Streamlit page
st.set_page_config(
    page_title="News Repository Dashboard",
    page_icon="📰",
    layout="wide"
)

# Create data directory if it doesn't exist
DATA_DIR = "news_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Create output directory for summaries if it doesn't exist
SUMMARIES_DIR = "news_summaries"
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# Create session state to store data
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# Header
st.title("News Repository Dashboard")
st.markdown("### Politics, Finance, Business & Economics News Repository")

# Sidebar controls
st.sidebar.header("Controls")

# Status indicators
st.sidebar.subheader("System Status")
st.sidebar.markdown("RSS Scraper: ✅ Available")

# Data loading options
st.sidebar.subheader("Data Sources")
sample_data_button = st.sidebar.button("Load Sample Data")
rss_fetch_button = st.sidebar.button("Fetch RSS News")

# Load previous data
st.sidebar.subheader("Load Previous Data")
saved_files = get_saved_news_files()
if saved_files:
    selected_file = st.sidebar.selectbox("Select saved data file", saved_files)
    load_file_button = st.sidebar.button("Load Selected File")
    
    if load_file_button:
        with st.spinner(f"Loading data from {selected_file}..."):
            st.session_state.news_data = load_news_file(selected_file)
            st.session_state.current_file = selected_file
            st.session_state.last_updated = datetime.now()
            st.success(f"Loaded {len(st.session_state.news_data)} news items from {selected_file}")
            st.rerun()
else:
    st.sidebar.info("No saved data files found")

# Button handlers
if sample_data_button:
    with st.spinner("Generating sample data..."):
        st.session_state.news_data = generate_sample_data()
        st.session_state.current_file = "sample_data"
        st.session_state.last_updated = datetime.now()
    st.success(f"Loaded {len(st.session_state.news_data)} sample news items")
    st.rerun()

if rss_fetch_button:
    with st.spinner("Fetching RSS news feeds..."):
        try:
            # Create RSS scraper and fetch data
            scraper = RSSNewsScraperMultiSource()
            st.info("Scraper created successfully. Fetching RSS feeds...")
            
            # Scrape all categories (now with no limit per feed)
            scraper.scrape_all_categories() 
            
            # Log the results
            st.info(f"Scraping completed. Found {len(scraper.all_articles)} articles.")
            
            # Remove duplicates
            scraper.remove_duplicates()
            st.info(f"After removing duplicates: {len(scraper.all_articles)} articles.")
            
            if scraper.all_articles:
                st.session_state.news_data = scraper.all_articles
                st.session_state.last_updated = datetime.now()
                
                # Save the data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"all_news_{timestamp}.csv"
                filepath = os.path.join(DATA_DIR, filename)
                
                df = pd.DataFrame(scraper.all_articles)
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                
                st.session_state.current_file = filename
                st.success(f"Successfully fetched {len(scraper.all_articles)} articles from RSS feeds")
            else:
                st.error("RSS fetch completed but no articles found")
                if st.session_state.news_data is None:
                    st.session_state.news_data = generate_sample_data()
                    st.session_state.current_file = "sample_data"
        except Exception as e:
            st.error(f"Error during RSS fetch: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            if st.session_state.news_data is None:
                st.session_state.news_data = generate_sample_data()
                st.session_state.current_file = "sample_data"
    st.rerun()

# Main dashboard content
if st.session_state.news_data is None or len(st.session_state.news_data) == 0:
    st.info("No data loaded. Please load sample data, fetch RSS news, or select a saved file.")
else:
    # Convert to DataFrame for processing
    df = pd.DataFrame(st.session_state.news_data)
    
    # Display current dataset info
    st.markdown(f"**Current dataset:** {st.session_state.current_file or 'None'}")
    if st.session_state.last_updated:
        st.markdown(f"**Last updated:** {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        st.metric("News Categories", df['category'].nunique())
    with col3:
        st.metric("News Sources", df['source'].nunique())
    with col4:
        # Calculate date range
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        date_range = (df['date'].max() - df['date'].min()).days + 1
        st.metric("Date Range (Days)", date_range)
    
    # Filters
    st.subheader("Filter News Articles")
    
    # Create columns for filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Date filter
        min_date = df['date'].min()
        max_date = df['date'].max()
        selected_date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
    with col2:
        # Category filter
        categories = sorted(df['category'].unique())
        selected_categories = st.multiselect("Categories", categories)
    
    with col3:
        # Source filter
        sources = sorted(df['source'].unique())
        selected_sources = st.multiselect("Sources", sources)
    
    with col4:
        # Search filter
        search_query = st.text_input("Search headlines or summaries")
    
    # Apply filters
    filtered_df = df.copy()
    
    # Date filter
    if len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Category filter
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    # Source filter
    if selected_sources:
        filtered_df = filtered_df[filtered_df['source'].isin(selected_sources)]
    
    # Search filter
    if search_query:
        search_query = search_query.lower()
        filtered_df = filtered_df[
            filtered_df['headline'].str.lower().str.contains(search_query, na=False) | 
            filtered_df['summary'].str.lower().str.contains(search_query, na=False)
        ]
    
    # Sort by date (newest first)
    filtered_df = filtered_df.sort_values('timestamp', ascending=False)
    
    # Create tabs for different views
    tabs = st.tabs(["Articles", "Charts & Trends", "AI Summaries"])
    
    # Tab 1: Articles (original table view)
    with tabs[0]:
        st.subheader(f"News Articles ({len(filtered_df)} results)")
        
        # Display the table
        st.dataframe(
            filtered_df[['timestamp', 'category', 'source', 'headline', 'summary', 'url']],
            column_config={
                "timestamp": "Date",
                "category": "Category",
                "source": "Source",
                "headline": "Headline",
                "summary": "Summary",
                "url": st.column_config.LinkColumn("Link")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download button for filtered data
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    
    # Tab 2: Charts & Trends
    with tabs[1]:
        st.subheader("News Analytics & Trends")
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            try:
                st.plotly_chart(create_category_chart(filtered_df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating category chart: {str(e)}")
        
        with col2:
            # Source distribution
            try:
                st.plotly_chart(create_source_chart(filtered_df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating source chart: {str(e)}")
        
        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Timeline of article volume
            try:
                st.plotly_chart(create_timeline_chart(filtered_df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating timeline chart: {str(e)}")
        
        with col2:
            # Keyword analysis for headlines
            try:
                st.plotly_chart(create_keyword_chart(filtered_df, column='headline'), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating keyword chart: {str(e)}")
        
        # Third row of charts
        st.subheader("Advanced Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Word cloud from headlines
            try:
                st.pyplot(create_wordcloud_figure(filtered_df, column='headline'))
            except Exception as e:
                st.error(f"Error creating word cloud: {str(e)}")
        
        with col2:
            # Word cloud from summaries
            try:
                st.pyplot(create_wordcloud_figure(filtered_df, column='summary'))
            except Exception as e:
                st.error(f"Error creating word cloud: {str(e)}")
        
        # Category-Source Heatmap
        try:
            st.plotly_chart(create_category_source_heatmap(filtered_df), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
    
    # Tab 3: AI Summaries
    with tabs[2]:
        st.subheader("AI-Generated News Summaries")
        
        # Claude API key handling with multiple sources and robust error handling
        if 'claude_api_key' not in st.session_state:
            # First check for GitHub secret/environment variable
            api_key = os.environ.get('CLAUDE_API_KEY', '')
            
            # If not found and dotenv is available, try reloading .env file
            if not api_key and 'load_dotenv' in globals():
                try:
                    # Reload to make sure we have the latest values
                    load_dotenv(override=True)
                    api_key = os.environ.get('CLAUDE_API_KEY', '')
                except Exception as e:
                    st.warning(f"Note: Could not load .env file: {e}")
            
            # If still not found, allow user input
            if not api_key:
                api_key = st.text_input(
                    "Enter your Claude API key:",
                    type="password",
                    help="Your API key will be stored only in this session and not saved to disk."
                )
                
            st.session_state.claude_api_key = api_key
        
        # Controls for summary generation
        col1, col2 = st.columns([3, 1])
        
        with col1:
            days_back = st.slider("Days back for summarization", 1, 7, 1)
            max_articles = st.slider("Max articles per category", 5, 30, 15)
        
        with col2:
            generate_button = st.button("Generate Summaries", type="primary")
        
        # Get articles from N days back
        target_df = get_previous_day_articles(filtered_df, days_back=days_back)
        
        # Show info about selected articles
        st.info(f"Found {len(target_df)} articles from {days_back} day(s) ago across {target_df['category'].nunique()} categories")
        
        # API key status indicator
        if st.session_state.claude_api_key:
            st.success("API key is configured")
        else:
            st.warning("No API key found. Please enter a Claude API key to generate summaries.")
        
        # Generate summaries when button is clicked
        if generate_button and not target_df.empty:
            if not st.session_state.claude_api_key:
                st.error("Please enter a Claude API key to generate summaries.")
            else:
                with st.spinner("Generating AI summaries... This may take a minute."):
                    try:
                        summaries = get_claude_summary(
                            target_df.to_dict('records'), 
                            st.session_state.claude_api_key,
                            max_articles_per_category=max_articles
                        )
                        
                        # Store summaries in session state
                        st.session_state.summaries = summaries
                        st.session_state.summary_timestamp = datetime.now()
                        
                        # Save summaries to file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        summary_file = os.path.join(SUMMARIES_DIR, f"news_summaries_{timestamp}.json")
                        with open(summary_file, 'w', encoding='utf-8') as f:
                            json.dump(summaries, f, indent=2)
                        
                        st.success(f"Summaries generated successfully and saved to {summary_file}!")
                    except Exception as e:
                        st.error(f"Error generating summaries: {str(e)}")
        
        # Display summaries if available
        if 'summaries' in st.session_state and st.session_state.summaries:
            # Show when summaries were generated
            if 'summary_timestamp' in st.session_state:
                st.caption(f"Summaries generated: {st.session_state.summary_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display each category summary
            for category, summary in st.session_state.summaries.items():
                with st.expander(f"{category} Summary", expanded=True):
                    st.markdown(summary)
            
            # Add button to download all summaries as JSON
            if st.button("Download All Summaries"):
                summary_json = json.dumps(st.session_state.summaries, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=summary_json,
                    file_name=f"news_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
        else:
            st.info("Click 'Generate Summaries' to create AI-powered analysis of recent news by category.")

# Add command-line argument support for headless operation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RSS News Dashboard')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no UI)')
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch and save news data')
    parser.add_argument('--generate-summaries', action='store_true', help='Generate summaries for latest data')
    
    args, unknown = parser.parse_known_args()
    
    # If running in headless mode
    if args.headless:
        if args.fetch_only:
            print("Running in headless mode: Fetching RSS feeds...")
            scraper = RSSNewsScraperMultiSource()
            scraper.scrape_all_categories()
            scraper.remove_duplicates()
            scraper.save_results()
            print(f"Saved {len(scraper.all_articles)} articles to disk.")
        
        if args.generate_summaries:
            print("Generating summaries for latest data...")
            # Find the latest data file
            data_files = get_saved_news_files()
            if not data_files:
                print("No data files found.")
                exit(1)
            
            latest_file = data_files[0]
            print(f"Using latest data file: {latest_file}")
            
            # Load the data
            articles = load_news_file(latest_file)
            
            # Get API key
            api_key = os.environ.get('CLAUDE_API_KEY')
            if not api_key:
                print("No Claude API key found in environment variables.")
                exit(1)
            
            # Generate summaries
            df = pd.DataFrame(articles)
            target_df = get_previous_day_articles(df, days_back=1)
            
            if target_df.empty:
                print("No articles found for the previous day.")
                exit(1)
            
            print(f"Generating summaries for {len(target_df)} articles...")
            summaries = get_claude_summary(
                target_df.to_dict('records'),
                api_key,
                max_articles_per_category=15
            )
            
            # Save summaries
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(SUMMARIES_DIR, f"news_summaries_{timestamp}.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2)
            
            print(f"Saved summaries to {summary_file}")
