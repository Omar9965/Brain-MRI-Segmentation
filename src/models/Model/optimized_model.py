import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import time
from typing import Optional, Union, Dict, Any, Tuple
from .unet import UNet
import onnx
import onnxruntime as ort
from pathlib import Path


class OptimizedModel:
    """Optimized model inference with multiple backends and quantization"""
    
    def __init__(self, model_path: str = None, use_quantization: bool = True, use_onnx: bool = True):
        self.model_path = model_path
        self.use_quantization = use_quantization
        self.use_onnx = use_onnx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model instances
        self.pytorch_model: Optional[UNet] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        self.quantized_model: Optional[UNet] = None
        
        # Performance tracking
        self.inference_times = []
        self.backend = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load different model backends based on configuration"""
        try:
            # Load PyTorch model
            if os.path.exists(self.model_path):
                self.pytorch_model = self._load_pytorch_model()
                
                # Create quantized version if enabled
                if self.use_quantization:
                    self.quantized_model = self._create_quantized_model()
                
                # Create ONNX version if enabled
                if self.use_onnx:
                    self._create_onnx_model()
                
                # Determine best backend
                self.backend = self._determine_best_backend()
                print(f"Model loaded successfully. Best backend: {self.backend}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def _load_pytorch_model(self) -> UNet:
        """Load PyTorch model"""
        model = UNet(n_classes=1, use_cbam=True)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _create_quantized_model(self) -> UNet:
        """Create quantized model for faster inference"""
        print("Creating quantized model...")
        model = self._load_pytorch_model()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        quantized_model.to(self.device)
        return quantized_model
    
    def _create_onnx_model(self):
        """Export model to ONNX format"""
        print("Exporting model to ONNX...")
        
        # Create a dummy input for export
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        
        # Export to ONNX
        onnx_path = os.path.splitext(self.model_path)[0] + ".onnx"
        torch.onnx.export(
            self.pytorch_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11
        )
        
        # Verify and load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        self.onnx_session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        print(f"ONNX model exported to: {onnx_path}")
    
    def _determine_best_backend(self) -> str:
        """Determine the best backend based on performance"""
        if self.onnx_session:
            return "onnx"
        elif self.quantized_model:
            return "quantized"
        else:
            return "pytorch"
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model inference"""
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input size (256x256)
        image = cv2.resize(image, (256, 256))
        
        # Normalize (same as training: mean=0.5, std=0.5)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        
        # Convert to tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image
    
    def _pytorch_inference(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run inference using PyTorch model"""
        with torch.no_grad():
            output = self.pytorch_model(input_tensor)
            pred_mask = (torch.sigmoid(output) > 0.5).float()
        return pred_mask.squeeze().cpu().numpy()
    
    def _quantized_inference(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run inference using quantized model"""
        with torch.no_grad():
            output = self.quantized_model(input_tensor)
            pred_mask = (torch.sigmoid(output) > 0.5).float()
        return pred_mask.squeeze().cpu().numpy()
    
    def _onnx_inference(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run inference using ONNX model"""
        # Convert to numpy and add batch dimension
        input_numpy = input_tensor.cpu().numpy()
        
        # Run inference
        outputs = self.onnx_session.run(None, {"input": input_numpy})
        pred_mask = (torch.sigmoid(torch.tensor(outputs[0])) > 0.5).float()
        return pred_mask.squeeze().numpy()
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run optimized inference"""
        start_time = time.time()
        
        # Preprocess image
        input_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        
        # Run inference based on selected backend
        if self.backend == "onnx":
            result = self._onnx_inference(input_tensor)
        elif self.backend == "quantized":
            result = self._quantized_inference(input_tensor)
        else:
            result = self._pytorch_inference(input_tensor)
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {"error": "No inference data available"}
        
        return {
            "backend": self.backend,
            "inference_times": self.inference_times,
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "total_inferences": len(self.inference_times)
        }
    
    def benchmark_backends(self, test_image: np.ndarray, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark different backends"""
        results = {}
        
        # Test PyTorch backend
        times = []
        for _ in range(num_runs):
            input_tensor = self.preprocess_image(test_image).unsqueeze(0).to(self.device)
            start_time = time.time()
            self._pytorch_inference(input_tensor)
            times.append(time.time() - start_time)
        results["pytorch"] = np.mean(times)
        
        # Test Quantized backend if available
        if self.quantized_model:
            times = []
            for _ in range(num_runs):
                input_tensor = self.preprocess_image(test_image).unsqueeze(0).to(self.device)
                start_time = time.time()
                self._quantized_inference(input_tensor)
                times.append(time.time() - start_time)
            results["quantized"] = np.mean(times)
        
        # Test ONNX backend if available
        if self.onnx_session:
            times = []
            for _ in range(num_runs):
                input_tensor = self.preprocess_image(test_image).unsqueeze(0).to(self.device)
                start_time = time.time()
                self._onnx_inference(input_tensor)
                times.append(time.time() - start_time)
            results["onnx"] = np.mean(times)
        
        return results


# Global optimized model instance
_optimized_model = None


def get_optimized_model(model_path: str = None, use_quantization: bool = True, use_onnx: bool = True) -> OptimizedModel:
    """Get or create the global optimized model instance"""
    global _optimized_model
    
    if _optimized_model is None:
        if model_path is None:
            # Default path
            model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        
        _optimized_model = OptimizedModel(
            model_path=model_path,
            use_quantization=use_quantization,
            use_onnx=use_onnx
        )
    
    return _optimized_model