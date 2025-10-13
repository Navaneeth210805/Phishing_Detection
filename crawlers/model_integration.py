"""
Integration script to connect crawlers with the phishing detection model
"""
import asyncio
import logging
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict

# Add backend directory to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from crawler_manager import CrawlerManager

try:
    from phishing_detection_system import PhishingDetectionSystem
except ImportError:
    # Fallback if main detection system is not available
    PhishingDetectionSystem = None

logger = logging.getLogger(__name__)

class CrawlerModelIntegration:
    """Integrates crawlers with phishing detection model"""
    
    def __init__(self):
        self.crawler_manager = CrawlerManager()
        self.detection_system = None
        self.results_file = "classified_urls.json"
        
        # Initialize detection system if available
        if PhishingDetectionSystem:
            try:
                self.detection_system = PhishingDetectionSystem()
                logger.info("Phishing detection system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize detection system: {e}")
    
    async def crawl_and_classify(self, crawler_name: str = None) -> List[Dict]:
        """Crawl URLs and classify them using the phishing detection model"""
        
        # Step 1: Crawl for URLs
        logger.info("Starting URL crawling...")
        urls = await self.crawler_manager.run_single_crawl(crawler_name)
        logger.info(f"Found {len(urls)} URLs from crawlers")
        
        if not urls:
            logger.info("No URLs found to classify")
            return []
        
        # Step 2: Classify URLs
        results = []
        if self.detection_system:
            logger.info("Classifying URLs with phishing detection model...")
            results = await self._classify_urls(urls)
        else:
            logger.warning("Detection system not available, saving URLs without classification")
            results = [{'url': url, 'classification': 'unknown'} for url in urls]
        
        # Step 3: Save results
        await self._save_classification_results(results)
        
        return results
    
    async def _classify_urls(self, urls: List[str]) -> List[Dict]:
        """Classify URLs using the phishing detection model"""
        results = []
        
        for i, url in enumerate(urls):
            try:
                logger.debug(f"Classifying URL {i+1}/{len(urls)}: {url}")
                
                # Use the detection system to classify the URL
                prediction = self.detection_system.predict_url(url)
                
                result = {
                    'url': url,
                    'is_phishing': prediction.get('is_phishing', False),
                    'confidence': prediction.get('confidence', 0.0),
                    'classification': 'phishing' if prediction.get('is_phishing') else 'legitimate',
                    'timestamp': datetime.now().isoformat(),
                    'features': prediction.get('features', {})
                }
                
                results.append(result)
                
                # Log high-confidence phishing detections
                if result['is_phishing'] and result['confidence'] > 0.8:
                    logger.warning(f"High-confidence phishing URL detected: {url} (confidence: {result['confidence']:.2f})")
                
            except Exception as e:
                logger.error(f"Error classifying URL {url}: {e}")
                results.append({
                    'url': url,
                    'classification': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    async def _save_classification_results(self, results: List[Dict]):
        """Save classification results to file"""
        try:
            # Load existing results
            existing_results = []
            if Path(self.results_file).exists():
                with open(self.results_file, 'r') as f:
                    existing_results = json.load(f)
            
            # Add new results
            existing_results.extend(results)
            
            # Keep only last 1000 results to prevent file from growing too large
            if len(existing_results) > 1000:
                existing_results = existing_results[-1000:]
            
            # Save back to file
            with open(self.results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            logger.info(f"Saved {len(results)} classification results to {self.results_file}")
            
        except Exception as e:
            logger.error(f"Error saving classification results: {e}")
    
    def get_classification_summary(self) -> Dict:
        """Get summary of classification results"""
        try:
            if not Path(self.results_file).exists():
                return {'error': 'No classification results found'}
            
            with open(self.results_file, 'r') as f:
                results = json.load(f)
            
            # Calculate statistics
            total = len(results)
            phishing_count = sum(1 for r in results if r.get('classification') == 'phishing')
            legitimate_count = sum(1 for r in results if r.get('classification') == 'legitimate')
            error_count = sum(1 for r in results if r.get('classification') == 'error')
            unknown_count = sum(1 for r in results if r.get('classification') == 'unknown')
            
            # Get high confidence phishing URLs
            high_confidence_phishing = [
                r for r in results 
                if r.get('is_phishing') and r.get('confidence', 0) > 0.8
            ]
            
            summary = {
                'total_urls': total,
                'phishing_urls': phishing_count,
                'legitimate_urls': legitimate_count,
                'error_count': error_count,
                'unknown_count': unknown_count,
                'phishing_percentage': (phishing_count / total * 100) if total > 0 else 0,
                'high_confidence_phishing': len(high_confidence_phishing),
                'last_update': max([r.get('timestamp', '') for r in results]) if results else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating classification summary: {e}")
            return {'error': str(e)}
    
    def get_phishing_urls(self, min_confidence: float = 0.5) -> List[Dict]:
        """Get URLs classified as phishing with minimum confidence"""
        try:
            if not Path(self.results_file).exists():
                return []
            
            with open(self.results_file, 'r') as f:
                results = json.load(f)
            
            phishing_urls = [
                r for r in results 
                if r.get('is_phishing') and r.get('confidence', 0) >= min_confidence
            ]
            
            # Sort by confidence (highest first)
            phishing_urls.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return phishing_urls
            
        except Exception as e:
            logger.error(f"Error getting phishing URLs: {e}")
            return []

async def main():
    """Main function for testing integration"""
    logging.basicConfig(level=logging.INFO)
    
    integration = CrawlerModelIntegration()
    
    print("Running crawler and classification integration...")
    results = await integration.crawl_and_classify()
    
    print(f"\nClassification completed! Processed {len(results)} URLs")
    
    # Show summary
    summary = integration.get_classification_summary()
    print("\n=== Classification Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Show high-confidence phishing URLs
    phishing_urls = integration.get_phishing_urls(min_confidence=0.8)
    if phishing_urls:
        print(f"\n=== High-Confidence Phishing URLs ({len(phishing_urls)}) ===")
        for result in phishing_urls[:10]:  # Show top 10
            print(f"URL: {result['url']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Timestamp: {result['timestamp']}")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
