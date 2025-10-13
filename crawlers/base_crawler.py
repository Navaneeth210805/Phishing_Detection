"""
Base crawler class that defines the interface for all crawlers
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
import aiohttp
import time

logger = logging.getLogger(__name__)

class BaseCrawler(ABC):
    """Base class for all crawlers"""
    
    def __init__(self, name: str, frequency_minutes: int, priority: str):
        self.name = name
        self.frequency_minutes = frequency_minutes
        self.priority = priority
        self.last_crawl_time = None
        self.session = None
        self.is_running = False
        
    async def init_session(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'PhishingDetectionBot/1.0 (Security Research)'
                }
            )
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    @abstractmethod
    async def crawl(self) -> List[str]:
        """
        Crawl for URLs and return a list of discovered URLs
        Returns:
            List[str]: List of URLs found
        """
        pass
    
    @abstractmethod
    def should_crawl(self) -> bool:
        """
        Check if enough time has passed since last crawl
        Returns:
            bool: True if crawler should run
        """
        pass
    
    def get_next_crawl_time(self) -> datetime:
        """Get the next scheduled crawl time"""
        if self.last_crawl_time:
            return datetime.fromtimestamp(
                self.last_crawl_time + (self.frequency_minutes * 60)
            )
        return datetime.now()
    
    def update_last_crawl_time(self):
        """Update the last crawl timestamp"""
        self.last_crawl_time = time.time()
    
    async def safe_crawl(self) -> List[str]:
        """
        Safely execute crawl with error handling
        Returns:
            List[str]: URLs found, empty list on error
        """
        try:
            logger.info(f"Starting crawl for {self.name}")
            await self.init_session()
            urls = await self.crawl()
            self.update_last_crawl_time()
            logger.info(f"Completed crawl for {self.name}, found {len(urls)} URLs")
            return urls
        except Exception as e:
            logger.error(f"Error in {self.name} crawler: {str(e)}")
            return []
        finally:
            await self.close_session()
    
    def __str__(self):
        return f"{self.name} (Priority: {self.priority}, Frequency: {self.frequency_minutes}min)"
