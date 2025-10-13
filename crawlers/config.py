"""
Configuration manager for API keys and settings
"""
import os
from pathlib import Path
from typing import Optional

class CrawlerConfig:
    """Manages configuration and API keys for crawlers"""
    
    def __init__(self):
        self._load_env_file()
    
    def _load_env_file(self):
        """Load environment variables from .env file if it exists"""
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    @property
    def google_api_key(self) -> Optional[str]:
        """Get Google Custom Search API key"""
        return os.getenv('GOOGLE_API_KEY')
    
    @property
    def google_search_engine_id(self) -> Optional[str]:
        """Get Google Custom Search Engine ID"""
        return os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    @property
    def phishtank_api_key(self) -> Optional[str]:
        """Get PhishTank API key"""
        return os.getenv('PHISHTANK_API_KEY')
    
    @property
    def urlscan_api_key(self) -> Optional[str]:
        """Get URLScan.io API key"""
        return os.getenv('URLSCAN_API_KEY')
    
    def has_google_config(self) -> bool:
        """Check if Google API is configured"""
        return bool(self.google_api_key and self.google_search_engine_id)
    
    def has_phishtank_config(self) -> bool:
        """Check if PhishTank API is configured"""
        return bool(self.phishtank_api_key)
    
    def has_urlscan_config(self) -> bool:
        """Check if URLScan.io API is configured"""
        return bool(self.urlscan_api_key)
    
    def get_config_status(self) -> dict:
        """Get status of all API configurations"""
        return {
            'google_search': {
                'configured': self.has_google_config(),
                'status': 'Ready' if self.has_google_config() else 'API key needed'
            },
            'phishtank': {
                'configured': self.has_phishtank_config(),
                'status': 'Ready' if self.has_phishtank_config() else 'Optional - using public API'
            },
            'urlscan': {
                'configured': self.has_urlscan_config(),
                'status': 'Ready' if self.has_urlscan_config() else 'Optional - public access only'
            }
        }

# Global config instance
config = CrawlerConfig()
