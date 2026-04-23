"""
Compare pgportal.png with example-pgportal.html using Visual Similarity Detector
This script uses the existing visual_similarity_detector.py module
"""
from visual_similarity_detector import (
    VisualPhishingDetector,
    ScreenshotCapture,
    DINOv2Comparator,
    PixelwiseComparator
)
from pathlib import Path
import os


def compare_pgportal():
    """
    Compare pgportal.png (reference image) with example-pgportal.html
    Uses all three comparison methods: DINOv2, Pixel-wise, and Combined
    """
    print("=" * 80)
    print("PG PORTAL VISUAL SIMILARITY COMPARISON")
    print("=" * 80)
    
    # Initialize detector
    print("\nInitializing Visual Phishing Detector...")
    detector = VisualPhishingDetector(
        model_name="facebook/dinov2-base",
        dinov2_weight=0.7,
        pixel_weight=0.3,
        rotation_invariant=True
    )
    
    # File paths
    png_path = "screenshots/pgportal.png"
    html_path = "example-pgportal.html"
    screenshot_dir = "./screenshots"
    
    # Ensure screenshot directory exists
    Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if PNG exists
    if not os.path.exists(png_path):
        print(f"\n❌ Error: {png_path} not found!")
        print(f"   Please ensure the file exists in the screenshots directory.")
        return
    
    # Check if HTML exists
    if not os.path.exists(html_path):
        print(f"\n❌ Error: {html_path} not found!")
        return
    
    print(f"\n✓ Found reference PNG: {png_path}")
    print(f"✓ Found HTML file: {html_path}")
    
    # Step 1: Capture HTML screenshot
    print("\n" + "-" * 80)
    print("Step 1: Capturing screenshot of example-pgportal.html")
    print("-" * 80)
    
    capturer = ScreenshotCapture()
    html_file_url = f"file://{Path(html_path).absolute()}"
    html_screenshot = f"{screenshot_dir}/example-pgportal-captured.png"

    print(f"URL: {html_file_url}")
    print("Capturing screenshot...")
    
    if not capturer.capture_screenshot(html_file_url, html_screenshot):
        print(f"❌ Failed to capture screenshot from HTML")
        return
    
    print(f"✓ Screenshot saved: {html_screenshot}")
    
    # Step 2: Compare using DINOv2 Only
    print("\n" + "=" * 80)
    print("COMPARISON METHOD 1: DINOv2 Semantic Similarity (Deep Learning)")
    print("=" * 80)
    print("Uses Vision Transformer to understand semantic content and layout")
    print("Good for: Detecting similar designs even with different text/colors\n")
    
    result_dinov2 = detector.compare_screenshots_dinov2(
        png_path,
        html_screenshot,
        reference_url="example-pgportal.html"
    )
    
    print(f"Reference:            {result_dinov2.reference_url}")
    print(f"Global Similarity:    {result_dinov2.global_similarity:.4f} (Overall semantic match)")
    print(f"Region Similarity:    {result_dinov2.region_similarity:.4f} (Important regions match)")
    print(f"Combined Score:       {result_dinov2.combined_score:.4f}")
    print(f"Verdict:              {result_dinov2.verdict}")
    print(f"Confidence:           {result_dinov2.confidence:.2%}")
    
    # Step 3: Compare using Pixel-wise Methods
    print("\n" + "=" * 80)
    print("COMPARISON METHOD 2: Pixel-wise Similarity (Traditional Computer Vision)")
    print("=" * 80)
    print("Uses SSIM, MSE, and histogram comparison")
    print("Good for: Exact visual matching, detecting pixel-level differences\n")
    
    result_pixel = detector.compare_screenshots_pixelwise(
        png_path,
        html_screenshot,
        reference_url="example-pgportal.html"
    )
    
    print(f"Reference:            {result_pixel.reference_url}")
    print(f"Pixel Similarity:     {result_pixel.pixel_similarity:.4f} (Combined pixel metrics)")
    print(f"SSIM Score:           {result_pixel.ssim_score:.4f} (Structural similarity)")
    print(f"Combined Score:       {result_pixel.combined_score:.4f}")
    print(f"Verdict:              {result_pixel.verdict}")
    print(f"Confidence:           {result_pixel.confidence:.2%}")
    
    if result_pixel.rotation_angle is not None:
        print(f"Rotation Detected:    {result_pixel.rotation_angle:.2f}°")
    
    # Step 4: Compare using Combined Method
    print("\n" + "=" * 80)
    print("COMPARISON METHOD 3: Combined (70% DINOv2 + 30% Pixel-wise)")
    print("=" * 80)
    print("Combines semantic understanding with pixel-level precision")
    print("Best for: Robust phishing detection with balanced accuracy\n")
    
    result_combined = detector.compare_screenshots(
        png_path,
        html_screenshot,
        reference_url="example-pgportal.html"
    )
    
    print(f"Reference:            {result_combined.reference_url}")
    print(f"Global Similarity:    {result_combined.global_similarity:.4f}")
    print(f"Region Similarity:    {result_combined.region_similarity:.4f}")
    print(f"Pixel Similarity:     {result_combined.pixel_similarity:.4f}")
    print(f"SSIM Score:           {result_combined.ssim_score:.4f}")
    print(f"Combined Score:       {result_combined.combined_score:.4f}")
    print(f"Verdict:              {result_combined.verdict}")
    print(f"Confidence:           {result_combined.confidence:.2%}")
    
    if result_combined.rotation_angle is not None:
        print(f"Rotation Detected:    {result_combined.rotation_angle:.2f}°")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nSimilarity Scores Comparison:")
    print(f"  DINOv2 Method:     {result_dinov2.combined_score:.4f}")
    print(f"  Pixel-wise Method: {result_pixel.combined_score:.4f}")
    print(f"  Combined Method:   {result_combined.combined_score:.4f}")
    
    print("\nInterpretation:")
    if result_combined.combined_score >= 0.95:
        print("  ✓ EXCELLENT match - Images are nearly identical")
        print("    The HTML renders very similarly to the reference PNG")
    elif result_combined.combined_score >= 0.85:
        print("  ✓ GOOD match - High visual similarity")
        print("    The HTML closely resembles the reference PNG")
    elif result_combined.combined_score >= 0.70:
        print("  ⚠ FAIR match - Some differences detected")
        print("    The HTML has noticeable differences from the reference PNG")
    else:
        print("  ✗ POOR match - Significant differences")
        print("    The HTML differs substantially from the reference PNG")
    
    print("\nIn phishing detection context:")
    if result_combined.verdict == "MATCHES_LEGITIMATE":
        print("  ⚠️ WARNING: High similarity to reference site detected!")
        print("  This could indicate a phishing attempt mimicking the legitimate site.")
    else:
        print("  ✓ Sites appear sufficiently different (likely safe)")
    
    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print(f"HTML Screenshot saved at: {html_screenshot}")
    print("=" * 80)
    
    # Optional: Generate attention visualization
    print("\n" + "-" * 80)
    print("Bonus: Generating DINOv2 Attention Visualization")
    print("-" * 80)
    print("This shows which regions DINOv2 considers important...\n")
    
    attention_output_png = f"{screenshot_dir}/pgportal_attention.png"
    attention_output_html = f"{screenshot_dir}/example-pgportal_attention.png"
    
    detector.dinov2.visualize_attention(png_path, attention_output_png)
    print(f"✓ PNG attention map: {attention_output_png}")
    
    detector.dinov2.visualize_attention(html_screenshot, attention_output_html)
    print(f"✓ HTML attention map: {attention_output_html}")
    
    return {
        'dinov2_result': result_dinov2,
        'pixel_result': result_pixel,
        'combined_result': result_combined,
        'html_screenshot': html_screenshot
    }


if __name__ == "__main__":
    try:
        results = compare_pgportal()
        print("\n✅ Script completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
