"""
Main crawler application
Provides command-line interface and integration with phishing detection model
"""
import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from crawler_manager import CrawlerManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PhishingCrawlerApp:
    """Main application for phishing detection crawlers"""
    
    def __init__(self):
        self.crawler_manager = None
        self.is_running = False
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def stop(self):
        """Stop the application"""
        self.is_running = False
        if self.crawler_manager:
            self.crawler_manager.stop_crawling()
    
    async def run_continuous(self):
        """Run crawlers continuously"""
        self.crawler_manager = CrawlerManager()
        self.is_running = True
        
        logger.info("Starting continuous crawling mode...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            await self.crawler_manager.start_continuous_crawling()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.is_running = False
    
    async def run_single_crawl(self, crawler_name=None):
        """Run a single crawl cycle"""
        self.crawler_manager = CrawlerManager()
        
        logger.info("Running single crawl...")
        urls = await self.crawler_manager.run_single_crawl(crawler_name)
        
        print(f"\nCrawl completed! Found {len(urls)} URLs:")
        for url in urls[:10]:  # Show first 10 URLs
            print(f"  - {url}")
        
        if len(urls) > 10:
            print(f"  ... and {len(urls) - 10} more URLs")
    
    def show_status(self):
        """Show crawler status"""
        self.crawler_manager = CrawlerManager()
        
        print("\n=== Crawler Status ===")
        status = self.crawler_manager.get_crawler_status()
        
        for crawler_status in status:
            print(f"\n{crawler_status['name']}:")
            print(f"  Priority: {crawler_status['priority']}")
            print(f"  Frequency: {crawler_status['frequency_minutes']} minutes")
            print(f"  Should crawl: {crawler_status['should_crawl']}")
            print(f"  Next crawl: {crawler_status['next_crawl']}")
            
            stats = crawler_status['stats']
            if stats:
                print(f"  Total URLs found: {stats.get('total_urls', 0)}")
                print(f"  Successful crawls: {stats.get('successful_crawls', 0)}")
                print(f"  Failed crawls: {stats.get('failed_crawls', 0)}")
    
    def show_stats(self):
        """Show crawler statistics"""
        self.crawler_manager = CrawlerManager()
        
        print("\n=== Crawler Statistics ===")
        stats = self.crawler_manager.get_stats()
        
        print(f"Total URLs found: {stats['total_urls_found']}")
        print(f"Crawls completed: {stats['crawls_completed']}")
        print(f"Last run: {stats['last_run'] or 'Never'}")
        
        print("\nPer-crawler stats:")
        for crawler_name, crawler_stats in stats['crawler_stats'].items():
            print(f"  {crawler_name}:")
            print(f"    URLs: {crawler_stats['total_urls']}")
            print(f"    Success: {crawler_stats['successful_crawls']}")
            print(f"    Failures: {crawler_stats['failed_crawls']}")
    
    def configure_crawler(self, crawler_name, frequency=None, priority=None):
        """Configure a specific crawler"""
        self.crawler_manager = CrawlerManager()
        
        config_updates = {}
        if frequency:
            config_updates['frequency_minutes'] = int(frequency)
        if priority:
            config_updates['priority'] = priority.upper()
        
        self.crawler_manager.configure_crawler(crawler_name, **config_updates)
        print(f"Updated configuration for {crawler_name}")
    
    def show_api_status(self):
        """Show API configuration status"""
        from config import config
        
        print("\n=== API Configuration Status ===")
        status = config.get_config_status()
        
        for service, info in status.items():
            status_icon = "✅" if info['configured'] else "❌"
            print(f"{status_icon} {service.replace('_', ' ').title()}: {info['status']}")
        
        print("\nTo configure API keys:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys to the .env file")
        print("3. Restart the crawler system")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Phishing Detection Crawler System')
    
    parser.add_argument('command', choices=[
        'run', 'single', 'status', 'stats', 'configure', 'api-status'
    ], help='Command to execute')
    
    parser.add_argument('--crawler', type=str, 
                       help='Specific crawler name for single crawl or configuration')
    
    parser.add_argument('--frequency', type=int,
                       help='Frequency in minutes for crawler configuration')
    
    parser.add_argument('--priority', choices=['HIGH', 'MEDIUM', 'LOW'],
                       help='Priority level for crawler configuration')
    
    args = parser.parse_args()
    
    app = PhishingCrawlerApp()
    app.setup_signal_handlers()
    
    try:
        if args.command == 'run':
            await app.run_continuous()
        
        elif args.command == 'single':
            await app.run_single_crawl(args.crawler)
        
        elif args.command == 'status':
            app.show_status()
        
        elif args.command == 'stats':
            app.show_stats()
        
        elif args.command == 'configure':
            if not args.crawler:
                print("Error: --crawler required for configure command")
                return 1
            
            app.configure_crawler(
                args.crawler, 
                frequency=args.frequency,
                priority=args.priority
            )
        
        elif args.command == 'api-status':
            app.show_api_status()
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
