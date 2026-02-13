"""
Visual Similarity Detector for Phishing Detection
Combines DINOv2 ViT (semantic) and pixel-wise comparison methods
Uses Playwright for screenshot capture
"""

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from playwright.sync_api import sync_playwright, Page, Browser
import asyncio
from playwright.async_api import async_playwright
import time


@dataclass
class SimilarityResult:
    """Result of similarity comparison"""
    reference_url: str
    global_similarity: float
    region_similarity: float
    pixel_similarity: float
    ssim_score: float
    combined_score: float
    verdict: str
    confidence: float
    rotation_angle: Optional[float] = None


class ScreenshotCapture:
    """Capture website screenshots using Playwright"""
    
    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Initialize screenshot capture
        
        Args:
            headless: Run browser in headless mode
            timeout: Page load timeout in milliseconds
        """
        self.headless = headless
        self.timeout = timeout
    
    def capture_screenshot(
        self, 
        url: str, 
        output_path: str,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        full_page: bool = False
    ) -> bool:
        """
        Capture screenshot of a website
        
        Args:
            url: Website URL to capture
            output_path: Path to save screenshot
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            full_page: Capture full scrollable page or just viewport
            
        Returns:
            Success status
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                context = browser.new_context(
                    viewport={'width': viewport_width, 'height': viewport_height}
                )
                page = context.new_page()
                page.set_default_timeout(self.timeout)
                
                # Navigate to URL
                page.goto(url, wait_until='networkidle')
                
                # Wait for page to be fully loaded
                page.wait_for_load_state('domcontentloaded')
                time.sleep(2)  # Additional wait for dynamic content
                
                # Capture screenshot
                page.screenshot(path=output_path, full_page=full_page)
                
                browser.close()
                return True
                
        except Exception as e:
            print(f"Error capturing screenshot for {url}: {str(e)}")
            return False
    
    def capture_multiple(
        self,
        urls: List[str],
        output_dir: str,
        prefix: str = "screenshot"
    ) -> List[str]:
        """
        Capture screenshots for multiple URLs
        
        Args:
            urls: List of URLs to capture
            output_dir: Directory to save screenshots
            prefix: Filename prefix
            
        Returns:
            List of saved screenshot paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, url in enumerate(urls):
            # Create safe filename from URL
            safe_name = url.replace('https://', '').replace('http://', '')
            safe_name = safe_name.replace('/', '_').replace(':', '_')
            filename = f"{prefix}_{i}_{safe_name}.png"
            filepath = output_path / filename
            
            if self.capture_screenshot(url, str(filepath)):
                saved_paths.append(str(filepath))
        
        return saved_paths


class RotationHandler:
    """Handles rotation detection and correction for screenshots"""
    
    @staticmethod
    def detect_rotation_orb(image1: np.ndarray, image2: np.ndarray) -> Optional[float]:
        """
        Detect rotation angle between two images using ORB feature matching
        
        Returns:
            Rotation angle in degrees, or None if detection fails
        """
        try:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
            
            orb = cv2.ORB_create(nfeatures=5000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return None
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) < 10:
                return None
            
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            M, inliers = cv2.estimateAffinePartial2D(pts1, pts2)
            
            if M is None:
                return None
            
            angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            return angle
            
        except Exception as e:
            print(f"Rotation detection failed: {str(e)}")
            return None
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle (degrees)
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (counter-clockwise)
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    @staticmethod
    def align_images(image1_path: str, image2_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """
        Align two images by detecting and correcting rotation
        
        Returns:
            Tuple of (aligned_image1, aligned_image2, detected_rotation_angle)
        """
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        angle = RotationHandler.detect_rotation_orb(img1, img2)
        
        if angle is not None and abs(angle) > 2.0:
            print(f"  Detected rotation: {angle:.2f}° - Correcting...")
            img2_aligned = RotationHandler.rotate_image(img2, -angle)
        else:
            img2_aligned = img2
            angle = 0.0 if angle is None else angle
        
        return img1, img2_aligned, angle


class PixelwiseComparator:
    """Traditional pixel-based comparison methods"""
    
    @staticmethod
    def compute_mse(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute Mean Squared Error between two images
        
        Returns:
            MSE score (lower is more similar)
        """
        mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
        return mse
    
    @staticmethod
    def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM)
        
        Returns:
            SSIM score (0-1, higher is more similar)
        """
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = image2
        
        # Compute mean
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variance and covariance
        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
        
        # Constants for stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Clamp to [0, 1] range
        ssim_score = float(np.mean(ssim_map))
        return max(0.0, min(1.0, ssim_score))
    
    @staticmethod
    def compute_histogram_similarity(
        image1: np.ndarray, 
        image2: np.ndarray
    ) -> float:
        """
        Compute histogram-based similarity
        
        Returns:
            Correlation score (0-1, higher is more similar)
        """
        # Compute histograms for each channel
        hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Compute correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(correlation)
    
    def compare(
        self, 
        image1_path: str, 
        image2_path: str,
        rotation_invariant: bool = True
    ) -> Dict[str, float]:
        """
        Compare two images using multiple pixel-based methods
        
        Args:
            rotation_invariant: If True, detect and correct rotation before comparison
            
        Returns:
            Dictionary with various similarity metrics
        """
        if rotation_invariant:
            img1, img2, rotation_angle = RotationHandler.align_images(image1_path, image2_path)
        else:
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            rotation_angle = None
        
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        mse = self.compute_mse(img1, img2)
        ssim = self.compute_ssim(img1, img2)
        hist_sim = self.compute_histogram_similarity(img1, img2)
        
        max_mse = 255.0 ** 2
        normalized_mse = 1.0 - min(mse / max_mse, 1.0)
        
        hist_sim = max(0.0, min(1.0, hist_sim))
        
        return {
            'mse': mse,
            'mse_normalized': normalized_mse,
            'ssim': ssim,
            'histogram_similarity': hist_sim,
            'pixel_similarity': (normalized_mse + ssim + hist_sim) / 3.0,
            'rotation_angle': rotation_angle
        }


class DINOv2Comparator:
    """DINOv2 Vision Transformer for semantic similarity"""
    
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        """
        Initialize DINOv2 model
        
        Args:
            model_name: HuggingFace model identifier
                - facebook/dinov2-small (22M params, faster)
                - facebook/dinov2-base (86M params, balanced)
                - facebook/dinov2-large (304M params, most accurate)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_features(
        self, 
        image_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract global and patch-level features from image
        
        Returns:
            cls_token: Global image representation (1, feature_dim)
            patch_tokens: Spatial features for regions (num_patches, feature_dim)
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # CLS token: global semantic representation
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Patch tokens: spatial features (exclude CLS token)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        
        return cls_token, patch_tokens
    
    def compute_global_similarity(
        self, 
        image1_path: str, 
        image2_path: str
    ) -> float:
        """
        Compute overall semantic similarity using CLS tokens
        
        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        cls1, _ = self.extract_features(image1_path)
        cls2, _ = self.extract_features(image2_path)
        
        similarity = F.cosine_similarity(cls1, cls2).item()
        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2
    
    def compute_region_similarity(
        self,
        image1_path: str,
        image2_path: str,
        top_k_patches: int = 50
    ) -> Dict[str, float]:
        """
        Compare important regions (headers, logos, forms) between images
        Focuses on patches with highest attention weights
        
        Args:
            top_k_patches: Number of most salient patches to compare
            
        Returns:
            Dictionary with region-based similarity metrics
        """
        _, patches1 = self.extract_features(image1_path)
        _, patches2 = self.extract_features(image2_path)
        
        # Compute attention weights (L2 norm indicates patch importance)
        attention1 = torch.norm(patches1, dim=-1)
        attention2 = torch.norm(patches2, dim=-1)
        
        # Get top-k most important patches from each image
        top_k = min(top_k_patches, patches1.size(1))
        _, top_idx1 = torch.topk(attention1.squeeze(), top_k)
        _, top_idx2 = torch.topk(attention2.squeeze(), top_k)
        
        important_patches1 = patches1.squeeze()[top_idx1]
        important_patches2 = patches2.squeeze()[top_idx2]
        
        # Global region similarity (average of important patches)
        global_region_sim = F.cosine_similarity(
            important_patches1.mean(dim=0, keepdim=True),
            important_patches2.mean(dim=0, keepdim=True)
        ).item()
        
        # Max similarity (best matching patch pairs)
        pairwise_sim = F.cosine_similarity(
            important_patches1.unsqueeze(1),
            important_patches2.unsqueeze(0),
            dim=-1
        )
        max_similarity = pairwise_sim.max(dim=1)[0].mean().item()
        
        # Convert from [-1, 1] to [0, 1]
        global_region_sim = (global_region_sim + 1) / 2
        max_similarity = (max_similarity + 1) / 2
        
        return {
            'global_region_similarity': global_region_sim,
            'max_patch_similarity': max_similarity,
            'region_similarity': (global_region_sim + max_similarity) / 2
        }
    
    def visualize_attention(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize which regions DINOv2 focuses on
        Useful for understanding what the model considers important
        
        Args:
            image_path: Input image path
            output_path: Optional path to save visualization
            
        Returns:
            Attention overlay as numpy array
        """
        image = Image.open(image_path).convert("RGB")
        _, patch_tokens = self.extract_features(image_path)
        
        # Compute attention map (L2 norm of patch features)
        attention = torch.norm(patch_tokens.squeeze(), dim=-1)
        
        # Reshape to 2D grid
        grid_size = int(np.sqrt(attention.shape[0]))
        attention_map = attention[:grid_size**2].reshape(grid_size, grid_size).cpu().numpy()
        
        # Resize to original image size
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        attention_resized = cv2.resize(attention_map, (w, h))
        
        # Normalize to 0-255
        attention_normalized = (attention_resized - attention_resized.min()) / \
                              (attention_resized.max() - attention_resized.min())
        heatmap = cv2.applyColorMap(
            (attention_normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Overlay on original image
        overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        return overlay


class VisualPhishingDetector:
    """
    Main phishing detector combining DINOv2 and pixel-wise methods
    Compares suspicious screenshots against database of legitimate sites
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        dinov2_weight: float = 0.7,
        pixel_weight: float = 0.3,
        rotation_invariant: bool = True
    ):
        """
        Initialize detector with both comparison methods
        
        Args:
            model_name: DINOv2 model variant
            dinov2_weight: Weight for semantic similarity (0-1)
            pixel_weight: Weight for pixel similarity (0-1)
            rotation_invariant: Enable rotation detection and correction
        """
        self.dinov2 = DINOv2Comparator(model_name)
        self.pixel = PixelwiseComparator()
        self.screenshot = ScreenshotCapture()
        self.rotation_invariant = rotation_invariant
        
        total = dinov2_weight + pixel_weight
        self.dinov2_weight = dinov2_weight / total
        self.pixel_weight = pixel_weight / total
    
    def compare_screenshots_dinov2(
        self,
        suspicious_path: str,
        reference_path: str,
        reference_url: str = "unknown"
    ) -> SimilarityResult:
        """
        Compare screenshots using DINOv2 semantic similarity only
        
        Args:
            suspicious_path: Path to suspicious website screenshot
            reference_path: Path to legitimate website screenshot
            reference_url: URL of reference site (for reporting)
            
        Returns:
            SimilarityResult with DINOv2 metrics and verdict
        """
        # DINOv2 semantic comparison
        global_sim = self.dinov2.compute_global_similarity(
            suspicious_path, 
            reference_path
        )
        region_metrics = self.dinov2.compute_region_similarity(
            suspicious_path,
            reference_path
        )
        
        combined_score = region_metrics['region_similarity']
        
        # Determine verdict: HIGH similarity means matches legitimate site
        if combined_score >= 0.85:
            verdict = "MATCHES_LEGITIMATE"
            confidence = combined_score
        elif combined_score >= 0.70:
            verdict = "POTENTIAL_PHISHING"
            confidence = combined_score
        else:
            verdict = "LIKELY_SAFE"
            confidence = 1.0 - combined_score
        
        return SimilarityResult(
            reference_url=reference_url,
            global_similarity=global_sim,
            region_similarity=region_metrics['region_similarity'],
            pixel_similarity=0.0,
            ssim_score=0.0,
            combined_score=combined_score,
            verdict=verdict,
            confidence=confidence
        )
    
    def compare_screenshots_pixelwise(
        self,
        suspicious_path: str,
        reference_path: str,
        reference_url: str = "unknown"
    ) -> SimilarityResult:
        """
        Compare screenshots using pixel-wise comparison only
        
        Args:
            suspicious_path: Path to suspicious website screenshot
            reference_path: Path to legitimate website screenshot
            reference_url: URL of reference site (for reporting)
            
        Returns:
            SimilarityResult with pixel-wise metrics and verdict
        """
        pixel_metrics = self.pixel.compare(
            suspicious_path, 
            reference_path,
            rotation_invariant=self.rotation_invariant
        )
        
        combined_score = pixel_metrics['pixel_similarity']
        
        if combined_score >= 0.85:
            verdict = "MATCHES_LEGITIMATE"
            confidence = combined_score
        elif combined_score >= 0.70:
            verdict = "POTENTIAL_PHISHING"
            confidence = combined_score
        else:
            verdict = "LIKELY_SAFE"
            confidence = 1.0 - combined_score
        
        return SimilarityResult(
            reference_url=reference_url,
            global_similarity=0.0,
            region_similarity=0.0,
            pixel_similarity=pixel_metrics['pixel_similarity'],
            ssim_score=pixel_metrics['ssim'],
            combined_score=combined_score,
            verdict=verdict,
            confidence=confidence,
            rotation_angle=pixel_metrics.get('rotation_angle')
        )
    
    def compare_screenshots(
        self,
        suspicious_path: str,
        reference_path: str,
        reference_url: str = "unknown"
    ) -> SimilarityResult:
        """
        Compare screenshots using combined DINOv2 and pixel-wise methods
        
        Args:
            suspicious_path: Path to suspicious website screenshot
            reference_path: Path to legitimate website screenshot
            reference_url: URL of reference site (for reporting)
            
        Returns:
            SimilarityResult with all metrics and verdict
        """
        global_sim = self.dinov2.compute_global_similarity(
            suspicious_path, 
            reference_path
        )
        region_metrics = self.dinov2.compute_region_similarity(
            suspicious_path,
            reference_path
        )
        
        pixel_metrics = self.pixel.compare(
            suspicious_path, 
            reference_path,
            rotation_invariant=self.rotation_invariant
        )
        
        combined_score = (
            self.dinov2_weight * region_metrics['region_similarity'] +
            self.pixel_weight * pixel_metrics['pixel_similarity']
        )
        
        if combined_score >= 0.85:
            verdict = "MATCHES_LEGITIMATE"
            confidence = combined_score
        elif combined_score >= 0.70:
            verdict = "POTENTIAL_PHISHING"
            confidence = combined_score
        else:
            verdict = "LIKELY_SAFE"
            confidence = 1.0 - combined_score
        
        return SimilarityResult(
            reference_url=reference_url,
            global_similarity=global_sim,
            region_similarity=region_metrics['region_similarity'],
            pixel_similarity=pixel_metrics['pixel_similarity'],
            ssim_score=pixel_metrics['ssim'],
            combined_score=combined_score,
            verdict=verdict,
            confidence=confidence,
            rotation_angle=pixel_metrics.get('rotation_angle')
        )
    
    def compare_with_database(
        self,
        suspicious_screenshot: str,
        legitimate_screenshots: List[str],
        legitimate_urls: Optional[List[str]] = None,
        threshold: float = 0.85
    ) -> Dict[str, any]:
        """
        Compare suspicious screenshot against entire database of legitimate sites
        
        Args:
            suspicious_screenshot: Path to suspicious screenshot
            legitimate_screenshots: List of paths to legitimate screenshots
            legitimate_urls: Optional list of URLs corresponding to screenshots
            threshold: Similarity threshold for phishing detection
            
        Returns:
            Detection results with best match and all comparisons
        """
        if legitimate_urls is None:
            legitimate_urls = [f"reference_{i}" for i in range(len(legitimate_screenshots))]
        
        results = []
        for screenshot, url in zip(legitimate_screenshots, legitimate_urls):
            result = self.compare_screenshots(
                suspicious_screenshot,
                screenshot,
                url
            )
            results.append(result)
        
        # Find best match (highest similarity)
        best_match = max(results, key=lambda x: x.combined_score)
        
        # Final verdict: HIGH similarity means it matches a legitimate site
        is_matching_legitimate = best_match.combined_score >= threshold
        
        return {
            'verdict': 'MATCHES_LEGITIMATE' if is_matching_legitimate else 'LIKELY_SAFE',
            'confidence': best_match.confidence,
            'best_match': {
                'url': best_match.reference_url,
                'combined_score': best_match.combined_score,
                'semantic_similarity': best_match.region_similarity,
                'pixel_similarity': best_match.pixel_similarity,
                'ssim': best_match.ssim_score
            },
            'all_comparisons': [
                {
                    'url': r.reference_url,
                    'combined_score': r.combined_score,
                    'verdict': r.verdict
                }
                for r in results
            ],
            'top_matches': sorted(
                results, 
                key=lambda x: x.combined_score, 
                reverse=True
            )[:5]
        }
    
    def detect_from_url(
        self,
        suspicious_url: str,
        legitimate_urls: List[str],
        screenshot_dir: str = "./screenshots"
    ) -> Dict[str, any]:
        """
        End-to-end detection: capture screenshots and compare
        
        Args:
            suspicious_url: URL of suspicious website
            legitimate_urls: List of legitimate website URLs
            screenshot_dir: Directory to save screenshots
            
        Returns:
            Detection results
        """
        screenshot_path = Path(screenshot_dir)
        screenshot_path.mkdir(parents=True, exist_ok=True)
        
        # Capture suspicious screenshot
        suspicious_file = screenshot_path / "suspicious.png"
        if not self.screenshot.capture_screenshot(suspicious_url, str(suspicious_file)):
            return {'error': 'Failed to capture suspicious screenshot'}
        
        # Capture or use existing legitimate screenshots
        legit_files = []
        for i, url in enumerate(legitimate_urls):
            legit_file = screenshot_path / f"legitimate_{i}.png"
            if not legit_file.exists():
                self.screenshot.capture_screenshot(url, str(legit_file))
            if legit_file.exists():
                legit_files.append(str(legit_file))
        
        # Compare
        return self.compare_with_database(
            str(suspicious_file),
            legit_files,
            legitimate_urls
        )


def test_rotation_invariance():
    """
    Test rotation handling by comparing original website with rotated version
    """
    print("\n" + "=" * 80)
    print("ROTATION INVARIANCE TEST")
    print("=" * 80)
    
    detector = VisualPhishingDetector(
        model_name="facebook/dinov2-base",
        dinov2_weight=0.7,
        pixel_weight=0.3,
        rotation_invariant=True
    )
    
    screenshot_dir = "./screenshots"
    Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n1. Capturing example.com screenshot...")
    capturer = ScreenshotCapture()
    original_screenshot = f"{screenshot_dir}/example_original.png"
    
    if capturer.capture_screenshot("http://example.com", original_screenshot):
        print(f"   Saved to: {original_screenshot}")
    else:
        print(f"   Failed to capture")
        return
    
    print("\n2. Capturing rotated example.com from HTML file...")
    rotated_url = f"file://{Path('/home/vk/Phishing_Detection/example_rotated.html').absolute()}"
    rotated_screenshot = f"{screenshot_dir}/example_rotated.png"
    
    if capturer.capture_screenshot(rotated_url, rotated_screenshot):
        print(f"   Saved to: {rotated_screenshot}")
    else:
        print(f"   Failed to capture")
        return
    
    print("\n" + "=" * 80)
    print("Comparing Original vs Rotated (15 degrees)")
    print("=" * 80)
    
    print("\n--- Using Combined Method (Rotation-Invariant) ---")
    result = detector.compare_screenshots(
        original_screenshot,
        rotated_screenshot,
        reference_url="example.com (rotated)"
    )
    
    if result.rotation_angle is not None:
        print(f"Detected Rotation:    {result.rotation_angle:.2f}°")
    print(f"Global Similarity:    {result.global_similarity:.4f}")
    print(f"Region Similarity:    {result.region_similarity:.4f}")
    print(f"Pixel Similarity:     {result.pixel_similarity:.4f}")
    print(f"SSIM Score:           {result.ssim_score:.4f}")
    print(f"Combined Score:       {result.combined_score:.4f}")
    print(f"Verdict:              {result.verdict}")
    print(f"Confidence:           {result.confidence:.2%}")
    
    print("\n--- Testing WITHOUT Rotation Correction ---")
    detector_no_rotation = VisualPhishingDetector(
        model_name="facebook/dinov2-base",
        dinov2_weight=0.7,
        pixel_weight=0.3,
        rotation_invariant=False
    )
    
    result_no_rot = detector_no_rotation.compare_screenshots(
        original_screenshot,
        rotated_screenshot,
        reference_url="example.com (rotated, no correction)"
    )
    
    print(f"Global Similarity:    {result_no_rot.global_similarity:.4f}")
    print(f"Region Similarity:    {result_no_rot.region_similarity:.4f}")
    print(f"Pixel Similarity:     {result_no_rot.pixel_similarity:.4f}")
    print(f"Combined Score:       {result_no_rot.combined_score:.4f}")
    print(f"Verdict:              {result_no_rot.verdict}")
    
    print("\n" + "=" * 80)
    print(f"Improvement with rotation correction: "
          f"{(result.combined_score - result_no_rot.combined_score) * 100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    print("Initializing Visual Phishing Detector...")
    
    # Test rotation invariance first
    test_rotation_invariance()
    
    # Original bank comparison tests
    detector = VisualPhishingDetector(
        model_name="facebook/dinov2-base",
        dinov2_weight=0.7,
        pixel_weight=0.3,
        rotation_invariant=True
    )
    
    print("\nCapturing screenshots and comparing...")
    print("=" * 80)
    
    # Test URLs
    sbi_url = "https://sbi.co.in/"
    icici_url = "https://www.icicibank.com/"
    
    # Capture screenshots
    screenshot_dir = "./screenshots"
    Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
    
    capturer = ScreenshotCapture()
    
    print(f"\n1. Capturing SBI Bank screenshot...")
    sbi_screenshot = f"{screenshot_dir}/sbi_bank.png"
    if capturer.capture_screenshot(sbi_url, sbi_screenshot):
        print(f"   Saved to: {sbi_screenshot}")
    else:
        print(f"   Failed to capture")
    
    print(f"\n2. Capturing ICICI Bank screenshot...")
    icici_screenshot = f"{screenshot_dir}/icici_bank.png"
    if capturer.capture_screenshot(icici_url, icici_screenshot):
        print(f"   Saved to: {icici_screenshot}")
    else:
        print(f"   Failed to capture")
    
    # Compare SBI with SBI (should be identical)
    print("\n" + "=" * 80)
    print("COMPARISON 1: SBI Bank vs SBI Bank (Same website)")
    print("=" * 80)
    
    print("\n--- Using DINOv2 Only ---")
    result_dinov2 = detector.compare_screenshots_dinov2(
        sbi_screenshot,
        sbi_screenshot,
        reference_url=sbi_url
    )
    print(f"Reference URL:        {result_dinov2.reference_url}")
    print(f"Global Similarity:    {result_dinov2.global_similarity:.4f}")
    print(f"Region Similarity:    {result_dinov2.region_similarity:.4f}")
    print(f"Combined Score:       {result_dinov2.combined_score:.4f}")
    print(f"Verdict:              {result_dinov2.verdict}")
    print(f"Confidence:           {result_dinov2.confidence:.2%}")
    
    print("\n--- Using Pixel-wise Only ---")
    result_pixel = detector.compare_screenshots_pixelwise(
        sbi_screenshot,
        sbi_screenshot,
        reference_url=sbi_url
    )
    print(f"Reference URL:        {result_pixel.reference_url}")
    print(f"Pixel Similarity:     {result_pixel.pixel_similarity:.4f}")
    print(f"SSIM Score:           {result_pixel.ssim_score:.4f}")
    print(f"Combined Score:       {result_pixel.combined_score:.4f}")
    print(f"Verdict:              {result_pixel.verdict}")
    print(f"Confidence:           {result_pixel.confidence:.2%}")
    
    print("\n--- Using Combined Method ---")
    result_combined = detector.compare_screenshots(
        sbi_screenshot,
        sbi_screenshot,
        reference_url=sbi_url
    )
    print(f"Reference URL:        {result_combined.reference_url}")
    print(f"Global Similarity:    {result_combined.global_similarity:.4f}")
    print(f"Region Similarity:    {result_combined.region_similarity:.4f}")
    print(f"Pixel Similarity:     {result_combined.pixel_similarity:.4f}")
    print(f"SSIM Score:           {result_combined.ssim_score:.4f}")
    print(f"Combined Score:       {result_combined.combined_score:.4f}")
    print(f"Verdict:              {result_combined.verdict}")
    print(f"Confidence:           {result_combined.confidence:.2%}")
    
    # Compare SBI with ICICI (different banks)
    print("\n" + "=" * 80)
    print("COMPARISON 2: SBI Bank vs ICICI Bank (Different websites)")
    print("=" * 80)
    
    print("\n--- Using DINOv2 Only ---")
    result_dinov2 = detector.compare_screenshots_dinov2(
        sbi_screenshot,
        icici_screenshot,
        reference_url=icici_url
    )
    print(f"Reference URL:        {result_dinov2.reference_url}")
    print(f"Global Similarity:    {result_dinov2.global_similarity:.4f}")
    print(f"Region Similarity:    {result_dinov2.region_similarity:.4f}")
    print(f"Combined Score:       {result_dinov2.combined_score:.4f}")
    print(f"Verdict:              {result_dinov2.verdict}")
    print(f"Confidence:           {result_dinov2.confidence:.2%}")
    
    print("\n--- Using Pixel-wise Only ---")
    result_pixel = detector.compare_screenshots_pixelwise(
        sbi_screenshot,
        icici_screenshot,
        reference_url=icici_url
    )
    print(f"Reference URL:        {result_pixel.reference_url}")
    print(f"Pixel Similarity:     {result_pixel.pixel_similarity:.4f}")
    print(f"SSIM Score:           {result_pixel.ssim_score:.4f}")
    print(f"Combined Score:       {result_pixel.combined_score:.4f}")
    print(f"Verdict:              {result_pixel.verdict}")
    print(f"Confidence:           {result_pixel.confidence:.2%}")
