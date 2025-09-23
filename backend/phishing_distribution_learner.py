#!/usr/bin/env python3
"""
Phishing Distribution Learner
=============================

This script learns the statistical distributions of features from known phishing URLs.
It extracts features from all phishing links and fits statistical distributions
to create a phishing profile that can be used for future detection.

The learned distributions are saved to disk for reuse in the detection system.
"""

import pandas as pd
import numpy as np
import json
import pickle
from scipy import stats
from scipy.stats import norm, gamma, beta, lognorm, expon, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import warnings
from phishing_feature_extractor import PhishingFeatureExtractor
import os
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhishingDistributionLearner:
    """Learn statistical distributions from phishing URLs."""
    
    def __init__(self, phishing_file: str = "../phishing_links.txt"):
        """Initialize the distribution learner."""
        self.phishing_file = phishing_file
        self.feature_extractor = PhishingFeatureExtractor()
        self.features_data = []
        self.distributions = {}
        self.feature_stats = {}
        
        # Distributions to test for continuous features
        self.continuous_distributions = [
            stats.norm,      # Normal distribution
            stats.gamma,     # Gamma distribution
            stats.beta,      # Beta distribution (for ratios)
            stats.lognorm,   # Log-normal distribution
            stats.expon,     # Exponential distribution
            stats.uniform,   # Uniform distribution
        ]
        
    def load_phishing_urls(self) -> List[str]:
        """Load phishing URLs from file."""
        logger.info(f"Loading phishing URLs from {self.phishing_file}")
        
        try:
            with open(self.phishing_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Loaded {len(urls)} phishing URLs")
            return urls
            
        except FileNotFoundError:
            logger.error(f"File {self.phishing_file} not found")
            return []
        except Exception as e:
            logger.error(f"Error loading URLs: {e}")
            return []
    
    def extract_features_from_urls(self, urls: List[str], max_urls: int = None) -> pd.DataFrame:
        """Extract features from all phishing URLs."""
        logger.info("Starting feature extraction from phishing URLs...")
        
        if max_urls:
            urls = urls[:max_urls]
            logger.info(f"Processing first {max_urls} URLs")
        
        all_features = []
        failed_count = 0
        
        for i, url in enumerate(urls, 1):
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(urls)} URLs...")
            
            try:
                # Extract all types of features
                features = {}
                
                # URL-based features (always available)
                url_features = self.feature_extractor.extract_url_features(url)
                features.update(url_features)
                
                # Try to get domain features (may fail for some URLs)
                try:
                    domain_features = self.feature_extractor.extract_domain_features(url)
                    features.update(domain_features)
                except Exception as e:
                    logger.debug(f"Domain feature extraction failed for {url}: {e}")
                
                # Try to get content features (may fail for some URLs)
                try:
                    content_features = self.feature_extractor.extract_content_features(url)
                    features.update(content_features)
                except Exception as e:
                    logger.debug(f"Content feature extraction failed for {url}: {e}")
                
                # Add metadata
                features['url'] = url
                features['is_phishing'] = 1  # All URLs in this file are phishing
                
                all_features.append(features)
                
            except Exception as e:
                logger.warning(f"Failed to extract features from {url}: {e}")
                failed_count += 1
                continue
        
        logger.info(f"Feature extraction completed. Success: {len(all_features)}, Failed: {failed_count}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        return df
    
    def clean_and_prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for distribution fitting."""
        logger.info("Cleaning and preparing features...")
        
        # Remove non-numeric columns for distribution analysis
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove URL and other identifier columns
        exclude_columns = ['url', 'domain_parts']
        numeric_features = [col for col in numeric_features if col not in exclude_columns]
        
        # Handle missing values
        df_clean = df[numeric_features].copy()
        
        # Fill missing values with median for continuous features
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        
        # Remove outliers using IQR method for better distribution fitting
        for col in df_clean.columns:
            if df_clean[col].dtype == 'float64':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them to preserve data
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Prepared {len(df_clean.columns)} numeric features: {list(df_clean.columns)}")
        return df_clean
    
    def fit_distribution(self, data: np.array, feature_name: str) -> Dict[str, Any]:
        """Fit the best distribution to the data."""
        # Remove any remaining NaN or infinite values
        data = data[np.isfinite(data)]
        
        if len(data) < 10:
            logger.warning(f"Not enough data points for {feature_name}: {len(data)}")
            return {
                'distribution': 'uniform',
                'parameters': [data.min(), data.max()],
                'ks_statistic': 1.0,
                'p_value': 0.0,
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data))
            }
        
        best_distribution = None
        best_params = None
        best_ks_stat = np.inf
        best_p_value = 0
        
        # Test each distribution
        for distribution in self.continuous_distributions:
            try:
                # Fit distribution
                if distribution == stats.beta:
                    # Beta distribution requires data to be in [0, 1]
                    if data.min() >= 0 and data.max() <= 1:
                        params = distribution.fit(data)
                    else:
                        continue
                else:
                    params = distribution.fit(data)
                
                # Perform Kolmogorov-Smirnov test
                ks_stat, p_value = stats.kstest(data, lambda x: distribution.cdf(x, *params))
                
                # Lower KS statistic indicates better fit
                if ks_stat < best_ks_stat:
                    best_distribution = distribution
                    best_params = params
                    best_ks_stat = ks_stat
                    best_p_value = p_value
                    
            except Exception as e:
                logger.debug(f"Failed to fit {distribution.name} to {feature_name}: {e}")
                continue
        
        if best_distribution is None:
            # Fallback to normal distribution
            best_distribution = stats.norm
            best_params = stats.norm.fit(data)
            best_ks_stat, best_p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, *best_params))
        
        return {
            'distribution': best_distribution.name,
            'parameters': list(best_params),
            'ks_statistic': float(best_ks_stat),
            'p_value': float(best_p_value),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75))
        }
    
    def learn_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Learn distributions for all features."""
        logger.info("Learning distributions for all features...")
        
        distributions = {}
        feature_stats = {}
        
        for feature in df.columns:
            logger.info(f"Fitting distribution for feature: {feature}")
            
            data = df[feature].values
            
            # Calculate basic statistics
            stats_info = {
                'count': int(len(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data))
            }
            
            feature_stats[feature] = stats_info
            
            # Determine if feature is binary/categorical or continuous
            unique_values = np.unique(data)
            
            if len(unique_values) <= 2:
                # Binary feature - use Bernoulli distribution
                p = np.mean(data)  # Probability of 1
                distributions[feature] = {
                    'type': 'binary',
                    'distribution': 'bernoulli',
                    'parameters': [float(p)],
                    'probability_1': float(p),
                    'probability_0': float(1 - p)
                }
                
            elif len(unique_values) <= 10 and np.all(data >= 0) and np.all(data == data.astype(int)):
                # Discrete feature - use Poisson or empirical distribution
                if np.max(data) <= 20:
                    # Poisson distribution
                    lambda_param = np.mean(data)
                    distributions[feature] = {
                        'type': 'discrete',
                        'distribution': 'poisson',
                        'parameters': [float(lambda_param)],
                        'values': unique_values.tolist(),
                        'probabilities': [float(np.mean(data == val)) for val in unique_values]
                    }
                else:
                    # Empirical distribution
                    value_counts = pd.Series(data).value_counts(normalize=True)
                    distributions[feature] = {
                        'type': 'empirical',
                        'distribution': 'empirical',
                        'values': value_counts.index.tolist(),
                        'probabilities': value_counts.values.tolist()
                    }
            else:
                # Continuous feature - fit best distribution
                dist_info = self.fit_distribution(data, feature)
                dist_info['type'] = 'continuous'
                distributions[feature] = dist_info
        
        self.distributions = distributions
        self.feature_stats = feature_stats
        
        logger.info(f"Learned distributions for {len(distributions)} features")
        return distributions
    
    def save_distributions(self, output_dir: str = "distributions"):
        """Save learned distributions to files."""
        logger.info(f"Saving distributions to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save distributions as JSON
        distributions_file = os.path.join(output_dir, "phishing_distributions.json")
        with open(distributions_file, 'w') as f:
            json.dump(self.distributions, f, indent=2)
        
        # Save feature statistics
        stats_file = os.path.join(output_dir, "phishing_feature_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.feature_stats, f, indent=2)
        
        # Save as pickle for Python use
        pickle_file = os.path.join(output_dir, "phishing_distributions.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'distributions': self.distributions,
                'feature_stats': self.feature_stats,
                'created_at': datetime.now().isoformat()
            }, f)
        
        logger.info(f"Distributions saved to:")
        logger.info(f"  - JSON: {distributions_file}")
        logger.info(f"  - Stats: {stats_file}")
        logger.info(f"  - Pickle: {pickle_file}")
    
    def generate_distribution_report(self, output_dir: str = "distributions"):
        """Generate a detailed report of learned distributions."""
        logger.info("Generating distribution report...")
        
        report_file = os.path.join(output_dir, "phishing_distribution_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("PHISHING URL DISTRIBUTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Features Analyzed: {len(self.distributions)}\n\n")
            
            # Summary by distribution type
            dist_types = {}
            for feature, dist_info in self.distributions.items():
                dist_type = dist_info.get('type', 'unknown')
                if dist_type not in dist_types:
                    dist_types[dist_type] = []
                dist_types[dist_type].append(feature)
            
            f.write("DISTRIBUTION TYPES SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for dist_type, features in dist_types.items():
                f.write(f"{dist_type.upper()}: {len(features)} features\n")
                for feature in features[:5]:  # Show first 5
                    f.write(f"  - {feature}\n")
                if len(features) > 5:
                    f.write(f"  ... and {len(features) - 5} more\n")
                f.write("\n")
            
            # Detailed feature analysis
            f.write("\nDETAILED FEATURE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for feature, dist_info in self.distributions.items():
                f.write(f"\nFeature: {feature}\n")
                f.write(f"Type: {dist_info.get('type', 'unknown')}\n")
                f.write(f"Distribution: {dist_info.get('distribution', 'unknown')}\n")
                
                if feature in self.feature_stats:
                    stats = self.feature_stats[feature]
                    f.write(f"Mean: {stats['mean']:.4f}\n")
                    f.write(f"Std: {stats['std']:.4f}\n")
                    f.write(f"Min: {stats['min']:.4f}\n")
                    f.write(f"Max: {stats['max']:.4f}\n")
                    f.write(f"Skewness: {stats['skewness']:.4f}\n")
                
                if 'ks_statistic' in dist_info:
                    f.write(f"KS Statistic: {dist_info['ks_statistic']:.4f}\n")
                    f.write(f"P-value: {dist_info['p_value']:.4f}\n")
                
                f.write("-" * 40 + "\n")
        
        logger.info(f"Distribution report saved to {report_file}")
    
    def plot_feature_distributions(self, df: pd.DataFrame, output_dir: str = "distributions", max_plots: int = 20):
        """Create visualization plots for feature distributions."""
        logger.info("Creating distribution plots...")
        
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Select features to plot (limit to avoid too many plots)
        features_to_plot = list(df.columns)[:max_plots]
        
        # Create individual plots
        for i, feature in enumerate(features_to_plot):
            plt.figure(figsize=(10, 6))
            
            data = df[feature].dropna()
            
            # Create subplot with histogram and fitted distribution
            plt.subplot(1, 2, 1)
            plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'{feature} - Histogram')
            plt.xlabel('Value')
            plt.ylabel('Density')
            
            # Plot fitted distribution if available
            if feature in self.distributions:
                dist_info = self.distributions[feature]
                if dist_info.get('type') == 'continuous':
                    try:
                        dist_name = dist_info['distribution']
                        params = dist_info['parameters']
                        
                        # Get the scipy distribution
                        dist = getattr(stats, dist_name)
                        
                        # Generate x values for plotting
                        x = np.linspace(data.min(), data.max(), 100)
                        y = dist.pdf(x, *params)
                        
                        plt.plot(x, y, 'r-', linewidth=2, label=f'Fitted {dist_name}')
                        plt.legend()
                    except Exception as e:
                        logger.debug(f"Could not plot distribution for {feature}: {e}")
            
            # Box plot
            plt.subplot(1, 2, 2)
            plt.boxplot(data)
            plt.title(f'{feature} - Box Plot')
            plt.ylabel('Value')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(plot_dir, f'{feature}_distribution.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create summary plot with multiple features
        plt.figure(figsize=(15, 10))
        
        # Select top 9 features for summary plot
        summary_features = features_to_plot[:9]
        
        for i, feature in enumerate(summary_features, 1):
            plt.subplot(3, 3, i)
            data = df[feature].dropna()
            plt.hist(data, bins=20, density=True, alpha=0.7, color='lightcoral')
            plt.title(f'{feature}', fontsize=10)
            plt.xlabel('Value', fontsize=8)
            plt.ylabel('Density', fontsize=8)
            plt.tick_params(labelsize=8)
        
        plt.tight_layout()
        summary_plot_file = os.path.join(plot_dir, 'feature_distributions_summary.png')
        plt.savefig(summary_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribution plots saved to {plot_dir}")
    
    def run_analysis(self, max_urls: int = None):
        """Run the complete distribution learning analysis."""
        logger.info("Starting phishing distribution learning analysis...")
        
        # Load URLs
        urls = self.load_phishing_urls()
        if not urls:
            logger.error("No URLs loaded. Exiting.")
            return
        
        # Extract features
        df = self.extract_features_from_urls(urls, max_urls)
        if df.empty:
            logger.error("No features extracted. Exiting.")
            return
        
        # Clean and prepare features
        df_clean = self.clean_and_prepare_features(df)
        
        # Learn distributions
        distributions = self.learn_distributions(df_clean)
        
        # Save results
        self.save_distributions()
        
        # Generate report
        self.generate_distribution_report()
        
        # Create plots
        self.plot_feature_distributions(df_clean)
        
        logger.info("Phishing distribution learning completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("PHISHING DISTRIBUTION LEARNING SUMMARY")
        print("="*60)
        print(f"URLs processed: {len(df)}")
        print(f"Features analyzed: {len(df_clean.columns)}")
        print(f"Distributions learned: {len(distributions)}")
        print("\nDistribution types:")
        
        dist_types = {}
        for feature, dist_info in distributions.items():
            dist_type = dist_info.get('type', 'unknown')
            dist_types[dist_type] = dist_types.get(dist_type, 0) + 1
        
        for dist_type, count in dist_types.items():
            print(f"  {dist_type}: {count} features")
        
        print(f"\nResults saved to: ./distributions/")
        print("="*60)


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Learn distributions from phishing URLs')
    parser.add_argument('--phishing-file', default='../phishing_links.txt', 
                       help='Path to phishing URLs file')
    parser.add_argument('--max-urls', type=int, default=None,
                       help='Maximum number of URLs to process (for testing)')
    parser.add_argument('--output-dir', default='distributions',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create learner
    learner = PhishingDistributionLearner(args.phishing_file)
    
    # Run analysis
    learner.run_analysis(max_urls=args.max_urls)


if __name__ == "__main__":
    main()
