import torch
import os
import cv2
import numpy as np
from typing import List, Optional, Union, Dict, Any
from .unet import UNet


# Paths
BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance (lazy loading)
_model: Optional[UNet] = None


def load_model(model_path: str = BEST_MODEL_PATH) -> UNet:
    """Load the U-Net model with trained weights."""
    global _model
    if _model is None:
        _model = UNet(n_classes=1, use_cbam=True)
        _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        _model.to(DEVICE)
        _model.eval()
    return _model




def preprocess_image(image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    """Preprocess image for model inference."""
    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_size = (image.shape[1], image.shape[0])
    
    # Resize to expected input size (256x256)
    image = cv2.resize(image, (256, 256))
    
    # Normalize (same as training: mean=0.5, std=0.5)
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    
    # Convert to tensor [C, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    
    return image, original_size


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG string."""
    _, buffer = cv2.imencode('.png', image)
    return buffer


def save_image(image: np.ndarray, filename: str) -> str:
    """Save image to output directory and return the URL path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, image)
    return f"/output/{filename}"


def create_overlay(original: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create an overlay of the mask on the original image."""
    # Ensure mask is same size as original
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # Create colored mask (red for tumor)
    colored_mask = np.zeros_like(original)
    colored_mask[:, :, 2] = mask_resized  # Red channel
    
    # Blend
    overlay = cv2.addWeighted(original, 1, colored_mask, alpha, 0)
    return overlay




def segment_image(
    image: Union[str, np.ndarray],
    filename: str = "image",
    model: Optional[UNet] = None,
    return_overlay: bool = True
) -> Dict[str, Any]:
    """
    Segment a single brain MRI image.
    
    Args:
        image: Either a file path (str) or numpy array (BGR format)
        filename: Original filename (without extension) for output naming
        model: Optional pre-loaded model instance
        return_overlay: Whether to include overlay visualization
    
    Returns:
        Dict with URLs to saved images, dimensions, and tumor detection status
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load image if path is provided
    if isinstance(image, str):
        original_image = cv2.imread(image)
        if original_image is None:
            raise ValueError(f"Could not load image: {image}")
    else:
        original_image = image.copy()
    
    original_h, original_w = original_image.shape[:2]
    
    # Preprocess
    input_tensor, _ = preprocess_image(original_image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (torch.sigmoid(output) > 0.5).float()
    
    # Convert mask to numpy
    mask_np = pred_mask.squeeze().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    
    # Resize mask back to original size
    mask_resized = cv2.resize(mask_np, (original_w, original_h))
    
    # Check if tumor is detected
    has_tumor = bool(np.any(mask_resized > 0))
    
    # Generate unique suffix for this segmentation
    import time
    timestamp = int(time.time() * 1000)
    base_name = f"{filename}_{timestamp}"
    
    # Save images to output directory
    original_url = save_image(original_image, f"{base_name}_original.png")
    mask_url = save_image(mask_resized, f"{base_name}_mask.png")
    
    # Create and save overlay if requested
    overlay_url = None
    if return_overlay:
        overlay = create_overlay(original_image, mask_resized)
        overlay_url = save_image(overlay, f"{base_name}_overlay.png")
    
    return {
        "filename": filename,
        "original_image_url": original_url,
        "mask_url": mask_url,
        "overlay_url": overlay_url,
        "width": original_w,
        "height": original_h,
        "has_tumor": has_tumor
    }


def segment_multiple_images(
    images: List[tuple[Union[str, np.ndarray], str]],
    return_overlay: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Segment multiple brain MRI images.
    
    Args:
        images: List of tuples (image_path_or_array, filename)
        return_overlay: Whether to include overlay visualizations
    
    Returns:
        Dict with 'results' list containing segmentation results
    """
    model = load_model()
    
    results = []
    for image, filename in images:
        result = segment_image(image, filename=filename, model=model, return_overlay=return_overlay)
        results.append(result)
    
    return {"results": results}

