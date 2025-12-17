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


def visualize_segmentation(
    result: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (15, 5)
) -> Optional[str]:
    """
    Visualize brain MRI segmentation result.
    
    Args:
        result: Dict with original_image, mask, overlay (base64), width, height, has_tumor
        save_path: Optional path to save the visualization
        show: Whether to display the plot
        figsize: Figure size for matplotlib
        
    Returns:
        Path to saved image if save_path is provided, else None
    """
    # Decode images from base64
    original = base64_to_numpy(result["original_image"])
    mask = base64_to_numpy(result["mask"])
    
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
    if result.get("overlay"):
        overlay = base64_to_numpy(result["overlay"])
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
        
        path = visualize_segmentation(result, save_path=save_path, show=show)
        if path:
            saved_paths.append(path)
    
    return saved_paths


def save_mask_image(
    result: Dict[str, Any],
    save_path: str
) -> str:
    """
    Save just the predicted mask as an image file.
    
    Args:
        result: Dict with mask (base64)
        save_path: Path to save the mask image
        
    Returns:
        Path to saved mask image
    """
    mask = base64_to_numpy(result["mask"])
    cv2.imwrite(save_path, mask)
    return save_path


def save_overlay_image(
    result: Dict[str, Any],
    save_path: str
) -> str:
    """
    Save the overlay visualization as an image file.
    
    Args:
        result: Dict with overlay or original_image and mask (base64)
        save_path: Path to save the overlay image
        
    Returns:
        Path to saved overlay image
    """
    if result.get("overlay"):
        overlay = base64_to_numpy(result["overlay"])
    else:
        # Create overlay manually
        original = base64_to_numpy(result["original_image"])
        mask = base64_to_numpy(result["mask"])
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
        
        # Create red overlay for tumor
        colored_mask = np.zeros_like(original)
        colored_mask[:, :, 2] = mask_gray
        overlay = cv2.addWeighted(original, 1, colored_mask, 0.4, 0)
    
    cv2.imwrite(save_path, overlay)
    return save_path

