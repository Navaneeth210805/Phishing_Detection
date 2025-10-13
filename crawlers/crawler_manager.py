"""
Crawler Manager
Orchestrates all crawlers according to their schedules and priorities
"""
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
import json
from pathlib import Path

from ct_logs_crawler import CTLogsCrawler
from typosquatting_crawler import TyposquattingCrawler
from google_search_crawler import GoogleSearchCrawler
from phishtank_crawler import PhishTankCrawler
from urlscan_crawler import URLScanCrawler
from config import config

logger = logging.getLogger(__name__)

class CrawlerManager:
    """Manages all crawlers and their execution schedules"""
    
    def __init__(self, config_file: str = None):
        self.crawlers = []
        self.is_running = False
        self.config_file = config_file or "crawler_config.json"
        self.results_file = "crawler_results.json"
        self.stats = {
            'total_urls_found': 0,
            'crawls_completed': 0,
            'last_run': None,
            'crawler_stats': {}
        }
        
        # Initialize crawlers
        self._initialize_crawlers()
        
        # Load configuration if exists
        self._load_config()
    
    def _initialize_crawlers(self):
        """Initialize all crawler instances"""
        try:
            # Initialize crawlers with API keys from config
            self.crawlers = [
                CTLogsCrawler(),
                TyposquattingCrawler(),
                GoogleSearchCrawler(
                    api_key=config.google_api_key,
                    search_engine_id=config.google_search_engine_id
                ),
                PhishTankCrawler(api_key=config.phishtank_api_key),
                URLScanCrawler(api_key=config.urlscan_api_key)
            ]
            
            # Initialize stats for each crawler
            for crawler in self.crawlers:
                self.stats['crawler_stats'][crawler.name] = {
                    'total_urls': 0,
                    'successful_crawls': 0,
                    'failed_crawls': 0,
                    'last_crawl': None
                }
            
            logger.info(f"Initialized {len(self.crawlers)} crawlers")
            
        except Exception as e:
            logger.error(f"Error initializing crawlers: {e}")
    
    def _load_config(self):
        """Load crawler configuration from file"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                # Apply configuration to crawlers
                for crawler_config in config.get('crawlers', []):
                    crawler_name = crawler_config.get('name')
                    crawler = self._get_crawler_by_name(crawler_name)
                    
                    if crawler:
                        # Update crawler settings
                        if 'frequency_minutes' in crawler_config:
                            crawler.frequency_minutes = crawler_config['frequency_minutes']
                        if 'priority' in crawler_config:
                            crawler.priority = crawler_config['priority']
                
                logger.info(f"Loaded configuration from {self.config_file}")
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def _save_config(self):
        """Save current crawler configuration"""
        try:
            config = {
                'crawlers': [
                    {
                        'name': crawler.name,
                        'frequency_minutes': crawler.frequency_minutes,
                        'priority': crawler.priority
                    }
                    for crawler in self.crawlers
                ]
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _get_crawler_by_name(self, name: str):
        """Get crawler instance by name"""
        for crawler in self.crawlers:
            if crawler.name == name:
                return crawler
        return None
    
    async def start_continuous_crawling(self):
        """Start continuous crawling with all crawlers"""
        logger.info("Starting continuous crawling...")
        self.is_running = True
        
        try:
            while self.is_running:
                # Check which crawlers should run
                active_crawlers = [
                    crawler for crawler in self.crawlers 
                    if crawler.should_crawl()
                ]
                
                if active_crawlers:
                    # Sort by priority (HIGH > MEDIUM > LOW)
                    priority_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                    active_crawlers.sort(
                        key=lambda c: priority_order.get(c.priority, 0),
                        reverse=True
                    )
                    
                    # Run crawlers
                    await self._run_crawlers(active_crawlers)
                
                # Wait before next check (check every minute)
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Crawler manager stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous crawling: {e}")
        finally:
            self.is_running = False
            await self._cleanup_crawlers()
    
    async def _run_crawlers(self, crawlers: List):
        """Run multiple crawlers concurrently"""
        logger.info(f"Running {len(crawlers)} crawlers: {[c.name for c in crawlers]}")
        
        # Create tasks for all crawlers
        tasks = []
        for crawler in crawlers:
            task = asyncio.create_task(self._run_single_crawler(crawler))
            tasks.append(task)
        
        # Wait for all crawlers to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_urls = []
        for i, result in enumerate(results):
            crawler = crawlers[i]
            
            if isinstance(result, Exception):
                logger.error(f"Crawler {crawler.name} failed: {result}")
                self.stats['crawler_stats'][crawler.name]['failed_crawls'] += 1
            elif isinstance(result, list):
                all_urls.extend(result)
                self.stats['crawler_stats'][crawler.name]['total_urls'] += len(result)
                self.stats['crawler_stats'][crawler.name]['successful_crawls'] += 1
                self.stats['crawler_stats'][crawler.name]['last_crawl'] = datetime.now().isoformat()
        
        # Update global stats
        self.stats['total_urls_found'] += len(all_urls)
        self.stats['crawls_completed'] += 1
        self.stats['last_run'] = datetime.now().isoformat()
        
        # Save results
        if all_urls:
            await self._save_results(all_urls)
        
        return all_urls
    
    async def _run_single_crawler(self, crawler) -> List[str]:
        """Run a single crawler safely"""
        try:
            urls = await crawler.safe_crawl()
            logger.info(f"{crawler.name} found {len(urls)} URLs")
            return urls
        except Exception as e:
            logger.error(f"Error running {crawler.name}: {e}")
            return []
    
    async def _save_results(self, urls: List[str]):
        """Save crawled URLs to file"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Load existing results
            existing_results = []
            if Path(self.results_file).exists():
                with open(self.results_file, 'r') as f:
                    existing_results = json.load(f)
            
            # Add new results
            new_result = {
                'timestamp': timestamp,
                'urls': urls,
                'count': len(urls)
            }
            
            existing_results.append(new_result)
            
            # Keep only last 100 crawl results to prevent file from growing too large
            if len(existing_results) > 100:
                existing_results = existing_results[-100:]
            
            # Save back to file
            with open(self.results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            logger.info(f"Saved {len(urls)} URLs to {self.results_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    async def _cleanup_crawlers(self):
        """Cleanup crawler resources"""
        for crawler in self.crawlers:
            try:
                await crawler.close_session()
            except Exception as e:
                logger.error(f"Error cleaning up {crawler.name}: {e}")
    
    def stop_crawling(self):
        """Stop the continuous crawling"""
        logger.info("Stopping crawler manager...")
        self.is_running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crawler statistics"""
        return self.stats.copy()
    
    def get_crawler_status(self) -> List[Dict[str, Any]]:
        """Get status of all crawlers"""
        status = []
        
        for crawler in self.crawlers:
            crawler_status = {
                'name': crawler.name,
                'priority': crawler.priority,
                'frequency_minutes': crawler.frequency_minutes,
                'should_crawl': crawler.should_crawl(),
                'next_crawl': crawler.get_next_crawl_time().isoformat() if crawler.last_crawl_time else 'Ready',
                'stats': self.stats['crawler_stats'].get(crawler.name, {})
            }
            status.append(crawler_status)
        
        return status
    
    async def run_single_crawl(self, crawler_name: str = None) -> List[str]:
        """Run a single crawl cycle"""
        if crawler_name:
            # Run specific crawler
            crawler = self._get_crawler_by_name(crawler_name)
            if crawler:
                return await self._run_crawlers([crawler])
            else:
                logger.error(f"Crawler '{crawler_name}' not found")
                return []
        else:
            # Run all crawlers that should crawl
            active_crawlers = [c for c in self.crawlers if c.should_crawl()]
            return await self._run_crawlers(active_crawlers)
    
    def configure_crawler(self, crawler_name: str, **kwargs):
        """Configure a specific crawler"""
        crawler = self._get_crawler_by_name(crawler_name)
        if crawler:
            for key, value in kwargs.items():
                if hasattr(crawler, key):
                    setattr(crawler, key, value)
                    logger.info(f"Updated {crawler_name}.{key} = {value}")
            
            # Save configuration
            self._save_config()
        else:
            logger.error(f"Crawler '{crawler_name}' not found")
