import os
import cv2
import base64
import numpy as np
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt


def base64_to_numpy(b64_string: str) -> np.ndarray:
    """Convert base64 encoded image string to numpy array."""
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image


def _load_image_from_result(result: Dict[str, Any], base64_key: str, url_key: str, output_dir: str = None) -> Optional[np.ndarray]:
    """Load an image from a result dict, supporting both base64 and URL/file-path formats.
    
    Args:
        result: Segmentation result dict
        base64_key: Key for base64-encoded image data (e.g., 'original_image')
        url_key: Key for URL/file-path image data (e.g., 'original_image_url')
        output_dir: Base directory to resolve relative URL paths against
        
    Returns:
        Loaded image as numpy array, or None if not found
    """
    # Try base64 format first
    if result.get(base64_key):
        return base64_to_numpy(result[base64_key])
    
    # Try URL/file-path format
    if result.get(url_key):
        url_path = result[url_key]
        
        # If it's a relative URL (e.g., /output/image.png), resolve to filesystem path
        if url_path.startswith("/output/") and output_dir:
            file_path = os.path.join(output_dir, os.path.basename(url_path))
        elif os.path.isabs(url_path):
            file_path = url_path
        else:
            # Try as relative path from project output directory
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(src_dir, "assets", "output", os.path.basename(url_path))
        
        if os.path.exists(file_path):
            return cv2.imread(file_path)
    
    return None


def visualize_segmentation(
    result: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (15, 5),
    output_dir: Optional[str] = None
) -> Optional[str]:
    """
    Visualize brain MRI segmentation result.
    
    Args:
        result: Dict with image data. Supports both base64 keys (original_image, mask, overlay)
                and URL keys (original_image_url, mask_url, overlay_url) from the web API.
        save_path: Optional path to save the visualization
        show: Whether to display the plot
        figsize: Figure size for matplotlib
        output_dir: Base directory to resolve URL paths. Defaults to project assets/output.
        
    Returns:
        Path to saved image if save_path is provided, else None
    """
    # Load images from result dict (supports both base64 and URL formats)
    original = _load_image_from_result(result, "original_image", "original_image_url", output_dir)
    mask = _load_image_from_result(result, "mask", "mask_url", output_dir)
    
    if original is None:
        raise ValueError("Could not load original image from result. Provide either 'original_image' (base64) or 'original_image_url'.")
    if mask is None:
        raise ValueError("Could not load mask from result. Provide either 'mask' (base64) or 'mask_url'.")
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")
    
    # Mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    
    # Overlay
    overlay = _load_image_from_result(result, "overlay", "overlay_url", output_dir)
    if overlay is not None:
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        axes[2].imshow(overlay_rgb)
    else:
        # Create overlay manually if not provided
        axes[2].imshow(original_rgb)
        axes[2].imshow(mask, cmap="jet", alpha=0.4)
    
    tumor_status = "Tumor Detected" if result.get("has_tumor") else "No Tumor"
    axes[2].set_title(f"Overlay ({tumor_status})")
    axes[2].axis("off")
    
    plt.suptitle(f"Brain MRI Segmentation - {result.get('width', 'N/A')}x{result.get('height', 'N/A')}", fontsize=14)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def visualize_multiple_segmentations(
    response: Dict[str, List[Dict[str, Any]]],
    output_dir: Optional[str] = None,
    show: bool = True
) -> List[str]:
    """
    Visualize multiple brain MRI segmentation results.
    
    Args:
        response: Dict with 'results' list containing segmentation results
        output_dir: Optional directory to save visualizations
        show: Whether to display plots
        
    Returns:
        List of saved file paths
    """
    saved_paths = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for idx, result in enumerate(response.get("results", [])):
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"segmentation_{idx}.png")
        
        path = visualize_segmentation(result, save_path=save_path, show=show, output_dir=output_dir)
        if path:
            saved_paths.append(path)
    
    return saved_paths


def save_mask_image(
    result: Dict[str, Any],
    save_path: str,
    output_dir: Optional[str] = None
) -> str:
    """
    Save just the predicted mask as an image file.
    
    Args:
        result: Dict with mask data (base64 or URL)
        save_path: Path to save the mask image
        output_dir: Base directory to resolve URL paths
        
    Returns:
        Path to saved mask image
    """
    mask = _load_image_from_result(result, "mask", "mask_url", output_dir)
    if mask is None:
        raise ValueError("Could not load mask from result.")
    cv2.imwrite(save_path, mask)
    return save_path


def save_overlay_image(
    result: Dict[str, Any],
    save_path: str,
    output_dir: Optional[str] = None
) -> str:
    """
    Save the overlay visualization as an image file.
    
    Args:
        result: Dict with overlay or original_image and mask data (base64 or URL)
        save_path: Path to save the overlay image
        output_dir: Base directory to resolve URL paths
        
    Returns:
        Path to saved overlay image
    """
    overlay = _load_image_from_result(result, "overlay", "overlay_url", output_dir)
    
    if overlay is None:
        # Create overlay manually
        original = _load_image_from_result(result, "original_image", "original_image_url", output_dir)
        mask = _load_image_from_result(result, "mask", "mask_url", output_dir)
        
        if original is None or mask is None:
            raise ValueError("Could not load original image and mask to create overlay.")
        
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
        
        # Create red overlay for tumor
        colored_mask = np.zeros_like(original)
        colored_mask[:, :, 2] = mask_gray
        overlay = cv2.addWeighted(original, 1, colored_mask, 0.4, 0)
    
    cv2.imwrite(save_path, overlay)
    return save_path


