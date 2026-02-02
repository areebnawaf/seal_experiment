"""Image Transformations for Robustness Evaluation"""

import io
import numpy as np
from PIL import Image, ImageFilter


class Transforms:
    """
    Image transformations for watermark robustness testing
    
    Implements the standard suite of transforms from the paper (Supplement §9.3):
    - Rotation (75 degrees)
    - JPEG compression (quality 25)
    - Crop and scale (75%)
    - Gaussian blur (8x8 kernel)
    - Gaussian noise (σ=0.1)
    - Brightness jitter (uniform 0-6)
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize transforms with random seed
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
    
    def apply(self, img: Image.Image, name: str) -> Image.Image:
        """
        Apply named transform to image
        
        Args:
            img: Input PIL Image
            name: Transform name (e.g., "Rotation_75")
            
        Returns:
            Transformed PIL Image
        """
        if name == "Original":
            return img.copy()
        
        elif name == "Rotation_75":
            return self._rotation(img, angle=75)
        
        elif name == "JPEG_25":
            return self._jpeg_compression(img, quality=25)
        
        elif name == "Crop_Scale_0.75":
            return self._crop_and_scale(img, scale=0.75)
        
        elif name == "Gaussian_Blur_8x8":
            return self._gaussian_blur(img, kernel_size=8)
        
        elif name == "Gaussian_Noise_0.1":
            return self._gaussian_noise(img, sigma=0.1)
        
        elif name == "Brightness_U06":
            return self._brightness_jitter(img, min_factor=0, max_factor=6)
        
        else:
            raise ValueError(f"Unknown transform: {name}")
    
    def _rotation(self, img: Image.Image, angle: float) -> Image.Image:
        """
        Rotate image by angle degrees
        
        Args:
            img: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        return img.rotate(angle, fillcolor='black')
    
    def _jpeg_compression(self, img: Image.Image, quality: int) -> Image.Image:
        """
        Apply JPEG compression at specified quality
        
        Args:
            img: Input image
            quality: JPEG quality (1-100, lower = more compression)
            
        Returns:
            JPEG compressed image
        """
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).copy()
    
    def _crop_and_scale(self, img: Image.Image, scale: float) -> Image.Image:
        """
        Randomly crop image to scale% and resize back to original size
        
        Args:
            img: Input image
            scale: Crop scale (0-1)
            
        Returns:
            Cropped and scaled image
        """
        w, h = img.size
        crop_w, crop_h = int(w * scale), int(h * scale)
        
        # Random crop position
        left = self.rng.randint(0, w - crop_w + 1)
        top = self.rng.randint(0, h - crop_h + 1)
        
        # Crop and resize back
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), Image.Resampling.LANCZOS)
    
    def _gaussian_blur(self, img: Image.Image, kernel_size: int) -> Image.Image:
        """
        Apply Gaussian blur
        
        Args:
            img: Input image
            kernel_size: Blur kernel size (radius = kernel_size/2)
            
        Returns:
            Blurred image
        """
        radius = kernel_size / 2
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _gaussian_noise(self, img: Image.Image, sigma: float) -> Image.Image:
        """
        Add Gaussian noise to image
        
        Args:
            img: Input image
            sigma: Noise standard deviation (0-1 scale)
            
        Returns:
            Noisy image
        """
        arr = np.array(img).astype(np.float32) / 255.0
        noise = self.rng.randn(*arr.shape) * sigma
        arr_noisy = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr_noisy * 255).astype(np.uint8))
    
    def _brightness_jitter(
        self,
        img: Image.Image,
        min_factor: float,
        max_factor: float
    ) -> Image.Image:
        """
        Apply random brightness adjustment
        
        Args:
            img: Input image
            min_factor: Minimum brightness factor
            max_factor: Maximum brightness factor
            
        Returns:
            Brightness-adjusted image
        """
        factor = self.rng.uniform(min_factor, max_factor)
        arr = np.array(img).astype(np.float32) * factor
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    @staticmethod
    def get_all_transforms() -> list:
        """
        Get list of all available transform names
        
        Returns:
            List of transform names
        """
        return [
            "Original",
            "Rotation_75",
            "JPEG_25",
            "Crop_Scale_0.75",
            "Gaussian_Blur_8x8",
            "Gaussian_Noise_0.1",
            "Brightness_U06"
        ]
