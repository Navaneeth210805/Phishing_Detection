# Phishing Detection Crawlers

This directory contains a comprehensive crawler system for discovering URLs in cyberspace to feed into the phishing detection model for classification.

## Overview

The crawler system implements an optimal crawling strategy with different priorities and frequencies:

| Source | Frequency | Priority | Description |
|--------|-----------|----------|-------------|
| CT Logs | Every 5-10 min | HIGH | Certificate Transparency logs monitoring |
| Typosquatting | Every 1 hour | HIGH | Generate and check typosquatting domains |
| Google Search | Every 30 min | MEDIUM | Search for suspicious sites |
| PhishTank | Every 15 min | MEDIUM | Fetch from PhishTank database |
| URLScan.io | Every 20 min | LOW | Public scan submissions |

## Architecture

### Core Components

1. **BaseCrawler** (`base_crawler.py`) - Abstract base class for all crawlers
2. **CrawlerManager** (`crawler_manager.py`) - Orchestrates all crawlers
3. **Individual Crawlers** - Specialized crawlers for each source
4. **Main Application** (`main.py`) - CLI interface and integration

### Crawler Implementations

#### CT Logs Crawler (`ct_logs_crawler.py`)
- Monitors Certificate Transparency logs for newly registered domains
- Focuses on suspicious patterns and brand impersonation
- Uses crt.sh API for certificate data
- **Priority**: HIGH, **Frequency**: 7 minutes

#### Typosquatting Crawler (`typosquatting_crawler.py`)
- Generates typosquatting variants of popular brands
- Checks if generated domains are active
- Uses character substitution, omission, and insertion techniques
- **Priority**: HIGH, **Frequency**: 60 minutes

#### Google Search Crawler (`google_search_crawler.py`)
- Uses Google Custom Search API to find suspicious sites
- Searches for phishing-related terms
- Requires API key configuration
- **Priority**: MEDIUM, **Frequency**: 30 minutes

#### PhishTank Crawler (`phishtank_crawler.py`)
- Fetches verified phishing URLs from PhishTank database
- Prioritizes recent and high-profile targets
- Uses PhishTank's public API
- **Priority**: MEDIUM, **Frequency**: 15 minutes

#### URLScan.io Crawler (`urlscan_crawler.py`)
- Monitors public URL scans for suspicious content
- Filters by malicious verdicts and suspicious patterns
- Optional API key for enhanced features
- **Priority**: LOW, **Frequency**: 20 minutes

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys (optional but recommended):
   - Google Custom Search API key and Search Engine ID
   - PhishTank API key for higher rate limits
   - URLScan.io API key for private scans

## Usage

### Command Line Interface

#### Start Continuous Crawling
```bash
python main.py run
```

#### Run Single Crawl
```bash
# All crawlers
python main.py single

# Specific crawler
python main.py single --crawler "CT Logs"
```

#### Check Status
```bash
python main.py status
```

#### View Statistics
```bash
python main.py stats
```

#### Configure Crawlers
```bash
# Change frequency
python main.py configure --crawler "CT Logs" --frequency 10

# Change priority
python main.py configure --crawler "Google Search" --priority HIGH
```

### Programmatic Usage

```python
from crawlers import CrawlerManager

# Initialize manager
manager = CrawlerManager()

# Run single crawl
urls = await manager.run_single_crawl()

# Start continuous crawling
await manager.start_continuous_crawling()

# Get status
status = manager.get_crawler_status()
stats = manager.get_stats()
```

## Configuration

### API Keys Configuration

Edit the crawler files to add your API keys:

```python
# Google Search Crawler
google_crawler = GoogleSearchCrawler(
    api_key="YOUR_GOOGLE_API_KEY",
    search_engine_id="YOUR_SEARCH_ENGINE_ID"
)

# PhishTank Crawler
phishtank_crawler = PhishTankCrawler(api_key="YOUR_PHISHTANK_API_KEY")

# URLScan.io Crawler
urlscan_crawler = URLScanCrawler(api_key="YOUR_URLSCAN_API_KEY")
```

### Crawler Settings

Settings are automatically saved to `crawler_config.json`:

```json
{
  "crawlers": [
    {
      "name": "CT Logs",
      "frequency_minutes": 7,
      "priority": "HIGH"
    }
  ]
}
```

## Output

### Results File
Crawled URLs are saved to `crawler_results.json`:

```json
[
  {
    "timestamp": "2025-10-08T10:30:00",
    "urls": [
      "https://secure-paypal-verify.com",
      "https://amazon-security-alert.net"
    ],
    "count": 2
  }
]
```

### Log File
Detailed logs are written to `crawler.log`:

```
2025-10-08 10:30:00 - CTLogsCrawler - INFO - Starting crawl for CT Logs
2025-10-08 10:30:05 - CTLogsCrawler - INFO - Completed crawl, found 15 URLs
```

## Integration with Phishing Detection Model

The crawlers are designed to work with the existing phishing detection system:

1. **URL Collection**: Crawlers discover potentially malicious URLs
2. **Model Classification**: URLs are fed to the phishing detection model
3. **Result Processing**: Classified URLs are stored and analyzed

### Integration Example

```python
# Add to backend/app.py or create new integration script
from crawlers import CrawlerManager
from phishing_detection_system import PhishingDetectionSystem

async def classify_crawled_urls():
    manager = CrawlerManager()
    detector = PhishingDetectionSystem()
    
    # Get URLs from crawlers
    urls = await manager.run_single_crawl()
    
    # Classify each URL
    results = []
    for url in urls:
        prediction = detector.predict_url(url)
        results.append({
            'url': url,
            'is_phishing': prediction['is_phishing'],
            'confidence': prediction['confidence']
        })
    
    return results
```

## Performance Considerations

- **Rate Limiting**: All crawlers implement rate limiting to avoid being blocked
- **Concurrent Requests**: Uses async/await for efficient I/O operations
- **Resource Management**: Automatic session cleanup and error handling
- **Result Caching**: Prevents duplicate URL processing

## Security Features

- **User-Agent Rotation**: Identifies as security research bot
- **Error Handling**: Graceful failure recovery
- **Timeout Management**: Prevents hanging requests
- **Suspicious Pattern Detection**: Multi-layer filtering for relevance

## Monitoring

The system provides comprehensive monitoring:

- **Real-time Status**: Current crawler states and schedules
- **Performance Metrics**: Success/failure rates and URL counts
- **Health Checks**: Automatic recovery from failures
- **Alerting**: Log-based monitoring for issues

## Extension Points

The modular design allows for easy extension:

1. **New Crawlers**: Inherit from `BaseCrawler`
2. **Custom Filters**: Add domain-specific filtering logic
3. **Additional Sources**: Integrate new threat intelligence feeds
4. **Enhanced Classification**: Add pre-filtering before model classification

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Increase delays or get higher-tier API access
2. **Network Timeouts**: Adjust timeout settings in base crawler
3. **Memory Usage**: Limit result set sizes and implement cleanup
4. **False Positives**: Tune filtering logic and suspicious patterns

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('crawlers').setLevel(logging.DEBUG)
```

## License

This crawler system is part of the Phishing Detection project and follows the same licensing terms.
