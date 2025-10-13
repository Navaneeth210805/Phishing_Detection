"""
Test script for the crawler system
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from crawler_manager import CrawlerManager
from ct_logs_crawler import CTLogsCrawler
from typosquatting_crawler import TyposquattingCrawler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_individual_crawlers():
    """Test individual crawlers"""
    print("=== Testing Individual Crawlers ===\n")
    
    # Test CT Logs Crawler
    print("1. Testing CT Logs Crawler...")
    ct_crawler = CTLogsCrawler()
    ct_urls = await ct_crawler.safe_crawl()
    print(f"   Found {len(ct_urls)} URLs from CT Logs")
    if ct_urls:
        print("   Sample URLs:")
        for url in ct_urls[:3]:
            print(f"     - {url}")
    print()
    
    # Test Typosquatting Crawler
    print("2. Testing Typosquatting Crawler...")
    typo_crawler = TyposquattingCrawler()
    typo_urls = await typo_crawler.safe_crawl()
    print(f"   Found {len(typo_urls)} URLs from Typosquatting")
    if typo_urls:
        print("   Sample URLs:")
        for url in typo_urls[:3]:
            print(f"     - {url}")
    print()

async def test_crawler_manager():
    """Test the crawler manager"""
    print("=== Testing Crawler Manager ===\n")
    
    manager = CrawlerManager()
    
    # Show crawler status
    print("Crawler Status:")
    status = manager.get_crawler_status()
    for crawler_status in status:
        print(f"  {crawler_status['name']}: {crawler_status['priority']} priority, "
              f"{crawler_status['frequency_minutes']} min frequency")
    print()
    
    # Run a single crawl
    print("Running single crawl cycle...")
    urls = await manager.run_single_crawl()
    print(f"Total URLs found: {len(urls)}")
    
    if urls:
        print("Sample URLs:")
        for url in urls[:5]:
            print(f"  - {url}")
    print()
    
    # Show statistics
    print("Crawler Statistics:")
    stats = manager.get_stats()
    print(f"  Total URLs found: {stats['total_urls_found']}")
    print(f"  Crawls completed: {stats['crawls_completed']}")
    print(f"  Last run: {stats['last_run']}")

def test_demo_functionality():
    """Test demo functionality without external APIs"""
    print("=== Demo Functionality ===\n")
    
    # Demonstrate typosquatting generation
    typo_crawler = TyposquattingCrawler()
    variants = typo_crawler._generate_typosquatting_variants("paypal")[:10]
    
    print("Sample typosquatting variants for 'paypal':")
    for variant in variants:
        print(f"  - {variant}")
    print()
    
    # Demonstrate suspicious domain detection
    ct_crawler = CTLogsCrawler()
    test_domains = [
        "paypal-verify.com",
        "amazon123.org", 
        "secure-microsoft.net",
        "google.com",
        "normal-website.com"
    ]
    
    print("Suspicious domain detection:")
    for domain in test_domains:
        is_suspicious = ct_crawler._is_suspicious_domain(domain)
        print(f"  {domain}: {'SUSPICIOUS' if is_suspicious else 'normal'}")

async def main():
    """Main test function"""
    print("Starting Crawler System Tests\n")
    print("=" * 50)
    
    # Test demo functionality first (no network required)
    test_demo_functionality()
    print()
    
    # Test individual crawlers
    await test_individual_crawlers()
    
    # Test crawler manager
    await test_crawler_manager()
    
    print("=" * 50)
    print("Test completed!")
    print("\nTo run the crawler system:")
    print("1. For continuous crawling: python main.py run")
    print("2. For single crawl: python main.py single")
    print("3. For status check: python main.py status")

if __name__ == "__main__":
    asyncio.run(main())
