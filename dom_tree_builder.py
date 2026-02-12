"""
DOM Tree Builder using Crawl4AI
Creates a tree structure from websites with unique IDs and parent-child relationships
Extracts all HTML features from a single BeautifulSoup parse
"""

from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import json
import asyncio
import re
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DOMNode:
    """Represents a single node in the DOM tree"""
    id: int                          # Unique identifier for this node
    tag: str                         # HTML tag name (div, p, a, etc.)
    parent_id: Optional[int]         # ID of parent node (None for root)
    child_ids: List[int]             # List of children IDs
    child_count: int                 # Number of direct children
    attributes: Dict[str, str]       # HTML attributes (class, id, href, etc.)
    text_content: str                # Text content (trimmed)
    depth: int                       # Depth in the tree (0 for root)
    

class DOMTreeBuilder:
    """Builds a DOM tree structure from a website URL"""
    
    def __init__(self):
        self.node_counter = 0
        self.dom_array = []
        self.soup = None  # Store parsed soup for feature extraction
        self.suspicious_keywords = ['verify', 'update', 'confirm', 'suspend', 'secure', 
                                   'account', 'login', 'password', 'urgent', 'alert']
        
    async def crawl_website(self, url: str, render_js: bool = True) -> str:
        """
        Crawls a website and returns HTML content with optional JavaScript rendering
        
        Args:
            url: Website URL to crawl
            render_js: Whether to execute JavaScript (default: True for dynamic content)
            
        Returns:
            HTML content as string (fully rendered if render_js=True)
        """
        async with AsyncWebCrawler(verbose=False) as crawler:
            if render_js:
                # Execute JavaScript to capture dynamically loaded content (jQuery, React, etc.)
                result = await crawler.arun(
                    url=url,
                    js_code=[
                        # Wait for jQuery and other frameworks to finish loading
                        """
                        await new Promise(resolve => {
                            if (typeof jQuery !== 'undefined') {
                                $(document).ready(resolve);
                            } else if (typeof React !== 'undefined' || typeof Vue !== 'undefined') {
                                setTimeout(resolve, 1000);
                            } else {
                                setTimeout(resolve, 500);
                            }
                        });
                        """,
                        # Additional wait for AJAX requests to complete
                        "await new Promise(resolve => setTimeout(resolve, 1500));"
                    ],
                    wait_for="body",
                    delay_before_return_html=2.0
                )
            else:
                # Fast mode: static HTML only (no JavaScript execution)
                result = await crawler.arun(url=url)
            
            return result.html
    
    async def build_tree_async(self, url: str, render_js: bool = True) -> List[Dict]:
        """
        Main method to build DOM tree from URL (async version)
        
        Args:
            url: Website URL to process
            render_js: Whether to execute JavaScript (True = full rendering, False = static only)
            
        Returns:
            List of DOMNode dictionaries representing the tree
        """
        # Reset for new tree
        self.node_counter = 0
        self.dom_array = []
        
        # Crawl and get HTML (with or without JS rendering)
        html_content = await self.crawl_website(url, render_js=render_js)
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Build tree starting from root
        self._build_node(soup.html if soup.html else soup, parent_id=None, depth=0)
        
        return [asdict(node) for node in self.dom_array]
    
    def build_tree(self, url: str, render_js: bool = True) -> List[Dict]:
        """
        Main method to build DOM tree from URL (synchronous wrapper)
        
        Args:
            url: Website URL to process
            render_js: Whether to execute JavaScript (True = captures jQuery/React/Vue content)
            
        Returns:
            List of DOMNode dictionaries representing the tree
        """
        return asyncio.run(self.build_tree_async(url, render_js=render_js))
    
    def _build_node(self, element, parent_id: Optional[int], depth: int) -> int:
        """
        Recursively builds DOM nodes from BeautifulSoup element
        
        Args:
            element: BeautifulSoup element to process
            parent_id: ID of parent node
            depth: Current depth in tree
            
        Returns:
            ID of created node
        """
        # Skip text nodes and comments
        if not hasattr(element, 'name') or element.name is None:
            return None
        
        # Create unique ID for this node
        current_id = self.node_counter
        self.node_counter += 1
        
        # Extract attributes (convert to dict, exclude None values)
        attributes = {}
        if hasattr(element, 'attrs'):
            for key, value in element.attrs.items():
                if isinstance(value, list):
                    attributes[key] = ' '.join(value)
                else:
                    attributes[key] = str(value)
        
        # Extract direct text content (not from children)
        text_content = ''
        if element.string:
            text_content = element.string.strip()
        
        # Placeholder for children (will be populated)
        child_ids = []
        
        # Create node
        node = DOMNode(
            id=current_id,
            tag=element.name,
            parent_id=parent_id,
            child_ids=child_ids,
            child_count=0,
            attributes=attributes,
            text_content=text_content,
            depth=depth
        )
        
        # Add to array
        self.dom_array.append(node)
        
        # Process children
        for child in element.children:
            # Skip NavigableString that are just whitespace
            if hasattr(child, 'name'):
                child_id = self._build_node(child, parent_id=current_id, depth=depth + 1)
                if child_id is not None:
                    child_ids.append(child_id)
        
        # Update child count
        node.child_count = len(child_ids)
        
        return current_id
    
    def get_node_by_id(self, node_id: int) -> Optional[Dict]:
        """
        Retrieve a node by its ID
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node dictionary or None if not found
        """
        for node in self.dom_array:
            if node.id == node_id:
                return asdict(node)
        return None
    
    def get_children(self, node_id: int) -> List[Dict]:
        """
        Get all children of a node
        
        Args:
            node_id: ID of parent node
            
        Returns:
            List of child node dictionaries
        """
        node = self.get_node_by_id(node_id)
        if not node:
            return []
        
        children = []
        for child_id in node['child_ids']:
            child = self.get_node_by_id(child_id)
            if child:
                children.append(child)
        return children
    
    def get_parent(self, node_id: int) -> Optional[Dict]:
        """
        Get parent of a node
        
        Args:
            node_id: ID of child node
            
        Returns:
            Parent node dictionary or None
        """
        node = self.get_node_by_id(node_id)
        if not node or node['parent_id'] is None:
            return None
        return self.get_node_by_id(node['parent_id'])
    
    def save_tree(self, filename: str):
        """
        Save DOM tree to JSON file
        
        Args:
            filename: Path to save file
        """
        tree_data = [asdict(node) for node in self.dom_array]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
    
    def load_tree(self, filename: str) -> List[Dict]:
        """
        Load DOM tree from JSON file
        
        Args:
            filename: Path to load file
            
        Returns:
            List of node dictionaries
        """
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def print_tree(self, node_id: int = 0, indent: int = 0):
        """
        Print tree structure in a readable format
        
        Args:
            node_id: Starting node ID (default root)
            indent: Current indentation level
        """
        node = self.get_node_by_id(node_id)
        if not node:
            return
        
        # Print current node
        prefix = "  " * indent
        tag_info = f"{node['tag']}"
        if node.get('attributes', {}).get('id'):
            tag_info += f" id='{node['attributes']['id']}'"
        if node.get('attributes', {}).get('class'):
            tag_info += f" class='{node['attributes']['class']}'"
        
        print(f"{prefix}[{node['id']}] {tag_info} (children: {node['child_count']})")
        
        # Print children recursively
        for child_id in node['child_ids']:
            self.print_tree(child_id, indent + 1)
    
    def extract_statistical_features(self) -> dict:
        """
        Extract statistical features from the built DOM tree
        Used for ML feature extraction without rebuilding the tree
        
        Returns:
            Dictionary with DOM structural statistics
        """
        if not self.dom_array:
            return {
                'max_dom_depth': 0,
                'avg_dom_depth': 0,
                'dom_balance': 0,
                'leaf_node_ratio': 0,
                'avg_children_per_node': 0,
                'dom_complexity': 0
            }
        
        depths = [node.depth for node in self.dom_array]
        leaf_nodes = sum(1 for node in self.dom_array if node.child_count == 0)
        child_counts = [node.child_count for node in self.dom_array]
        
        max_depth = max(depths)
        avg_depth = np.mean(depths)
        balance = np.std(depths) if len(depths) > 1 else 0
        
        return {
            'max_dom_depth': max_depth,
            'avg_dom_depth': float(avg_depth),
            'dom_balance': float(balance),
            'leaf_node_ratio': leaf_nodes / len(self.dom_array),
            'avg_children_per_node': float(np.mean(child_counts)),
            'dom_complexity': (len(self.dom_array) * max_depth) / (balance + 1)
        }
    
    def build_from_html_string(self, html: str):
        """
        Build DOM tree directly from HTML string (without URL crawling)
        Useful for ML feature extraction from existing HTML
        
        Args:
            html: Raw HTML string
        """
        # Reset for new tree
        self.node_counter = 0
        self.dom_array = []
        
        # Parse HTML and store soup for feature extraction
        self.soup = BeautifulSoup(html, 'html.parser')
        
        # Build tree starting from root
        self._build_node(self.soup.html if self.soup.html else self.soup, parent_id=None, depth=0)
    
    def extract_all_html_features(self) -> Dict:
        """
        Extract ALL HTML features from the parsed soup and DOM tree
        Single source of truth for HTML feature extraction
        Combines: tag counts, DOM structure, suspicious patterns
        
        Returns:
            Dictionary with all HTML features (22 features total)
        """
        if self.soup is None or not self.dom_array:
            # Return zero features if tree not built
            return self._get_zero_features()
        
        features = {}
        
        try:
            # === TAG COUNTING FEATURES (from soup) ===
            features['num_tags'] = len(self.soup.find_all())
            features['num_forms'] = len(self.soup.find_all('form'))
            features['num_inputs'] = len(self.soup.find_all('input'))
            features['num_links'] = len(self.soup.find_all('a'))
            features['num_scripts'] = len(self.soup.find_all('script'))
            features['num_iframes'] = len(self.soup.find_all('iframe'))
            features['num_images'] = len(self.soup.find_all('img'))
            features['num_divs'] = len(self.soup.find_all('div'))
            
            # Password field detection
            password_inputs = self.soup.find_all('input', {'type': 'password'})
            features['num_password_fields'] = len(password_inputs)
            
            # External links analysis
            links = self.soup.find_all('a', href=True)
            external_links = 0
            for link in links:
                href = link['href']
                if href.startswith('http') and not href.startswith('#'):
                    external_links += 1
            features['num_external_links'] = external_links
            features['external_link_ratio'] = external_links / max(len(links), 1)
            
            # Text analysis
            text = self.soup.get_text().lower()
            features['suspicious_keyword_count'] = sum(1 for kw in self.suspicious_keywords if kw in text)
            
            # Meta tags
            features['has_meta_description'] = 1 if self.soup.find('meta', {'name': 'description'}) else 0
            features['has_meta_keywords'] = 1 if self.soup.find('meta', {'name': 'keywords'}) else 0
            
            # Title
            title = self.soup.find('title')
            features['title_length'] = len(title.text) if title else 0
            
            # Hidden elements
            hidden_elements = self.soup.find_all(style=re.compile(r'display:\s*none|visibility:\s*hidden'))
            features['num_hidden_elements'] = len(hidden_elements)
            
            # === DOM TREE STRUCTURAL FEATURES ===
            dom_stats = self.extract_statistical_features()
            features.update(dom_stats)
            
        except Exception as e:
            print(f"Error extracting HTML features: {e}")
            features = self._get_zero_features()
        
        return features
    
    def _get_zero_features(self) -> Dict:
        """Return zero-valued features when extraction fails"""
        return {
            'num_tags': 0, 'num_forms': 0, 'num_inputs': 0, 'num_links': 0,
            'num_scripts': 0, 'num_iframes': 0, 'num_images': 0, 'num_divs': 0,
            'num_password_fields': 0, 'num_external_links': 0, 'external_link_ratio': 0,
            'suspicious_keyword_count': 0, 'has_meta_description': 0, 'has_meta_keywords': 0,
            'title_length': 0, 'num_hidden_elements': 0,
            'max_dom_depth': 0, 'avg_dom_depth': 0, 'dom_balance': 0,
            'leaf_node_ratio': 0, 'avg_children_per_node': 0, 'dom_complexity': 0
        }


class DOMTreeComparator:
    """Compare two DOM trees for similarity"""
    
    @staticmethod
    def compare_structure(tree1: List[Dict], tree2: List[Dict]) -> Dict:
        """
        Compare two DOM trees
        
        Args:
            tree1: First DOM tree
            tree2: Second DOM tree
            
        Returns:
            Dictionary with comparison metrics
        """
        # Basic structural metrics
        metrics = {
            'tree1_nodes': len(tree1),
            'tree2_nodes': len(tree2),
            'size_difference': abs(len(tree1) - len(tree2)),
            'tag_distribution_tree1': DOMTreeComparator._get_tag_distribution(tree1),
            'tag_distribution_tree2': DOMTreeComparator._get_tag_distribution(tree2),
            'max_depth_tree1': max([node['depth'] for node in tree1]) if tree1 else 0,
            'max_depth_tree2': max([node['depth'] for node in tree2]) if tree2 else 0,
        }
        
        return metrics
    
    @staticmethod
    def _get_tag_distribution(tree: List[Dict]) -> Dict[str, int]:
        """Get count of each tag type in tree"""
        distribution = {}
        for node in tree:
            tag = node['tag']
            distribution[tag] = distribution.get(tag, 0) + 1
        return distribution
    
    @staticmethod
    def find_similar_nodes(tree1: List[Dict], tree2: List[Dict], 
                          similarity_threshold: float = 0.8) -> List[tuple]:
        """
        Find similar nodes between two trees based on tag and attributes
        
        Args:
            tree1: First DOM tree
            tree2: Second DOM tree
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (node1_id, node2_id, similarity_score) tuples
        """
        similar_pairs = []
        
        for node1 in tree1:
            for node2 in tree2:
                score = DOMTreeComparator._node_similarity(node1, node2)
                if score >= similarity_threshold:
                    similar_pairs.append((node1['id'], node2['id'], score))
        
        return similar_pairs
    
    @staticmethod
    def _node_similarity(node1: Dict, node2: Dict) -> float:
        """
        Calculate similarity score between two nodes
        
        Returns:
            Similarity score between 0 and 1
        """
        score = 0.0
        
        # Tag match (40% weight)
        if node1['tag'] == node2['tag']:
            score += 0.4
        
        # Child count similarity (20% weight)
        if node1['child_count'] == node2['child_count']:
            score += 0.2
        elif node1['child_count'] > 0 and node2['child_count'] > 0:
            ratio = min(node1['child_count'], node2['child_count']) / max(node1['child_count'], node2['child_count'])
            score += 0.2 * ratio
        
        # Attribute similarity (40% weight)
        attrs1 = set(node1.get('attributes', {}).keys())
        attrs2 = set(node2.get('attributes', {}).keys())
        
        if attrs1 or attrs2:
            intersection = len(attrs1 & attrs2)
            union = len(attrs1 | attrs2)
            score += 0.4 * (intersection / union if union > 0 else 0)
        
        return score


# Example usage and testing
if __name__ == "__main__":
    # Create builder
    builder = DOMTreeBuilder()
    
    print("Building DOM tree from sbi.bank.in...")
    tree = builder.build_tree("https://sbi.bank.in/",render_js=True)
    
    print(f"\nTotal nodes: {len(tree)}")
    print(f"Root node: {tree[0] if tree else 'None'}")
    
    # Print tree structure
    print("\n--- Tree Structure ---")
    builder.print_tree()
    
    # Save tree
    builder.save_tree("example_dom_tree.json")
    print("\nTree saved to example_dom_tree.json")
    
    # print("\n\n=== Comparing Two Websites ===")
    # builder1 = DOMTreeBuilder()
    # builder2 = DOMTreeBuilder()
    
    # tree1 = builder1.build_tree("https://sbi.bank.in/")
    # tree2 = builder2.build_tree("https://example.org")
    
    # # Compare
    # comparator = DOMTreeComparator()
    # metrics = comparator.compare_structure(tree1, tree2)
    
    # print(f"\nComparison Metrics:")
    # print(f"  Tree 1 nodes: {metrics['tree1_nodes']}")
    # print(f"  Tree 2 nodes: {metrics['tree2_nodes']}")
    # print(f"  Size difference: {metrics['size_difference']}")
    # print(f"  Max depth tree1: {metrics['max_depth_tree1']}")
    # print(f"  Max depth tree2: {metrics['max_depth_tree2']}")
    
    # # Find similar nodes
    # similar = comparator.find_similar_nodes(tree1, tree2, similarity_threshold=0.7)
    # print(f"\nFound {len(similar)} similar node pairs")
    # if similar:
    #     print("Sample matches:")
    #     for node1_id, node2_id, score in similar[:5]:
    #         print(f"  Node {node1_id} <-> Node {node2_id} (score: {score:.2f})")
