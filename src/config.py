from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    captioner_model: str = "Salesforce/blip2-flan-t5-xl"
    embedder_model: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    
    n_patches: int = 1024
    bits_per_patch: int = 7
    patch_size: int = 2
    tau: float = 2.3
    
    gen_steps: int = 50
    inv_steps: int = 50
    guidance_scale: float = 7.5
    
    n_search_captions: int = 16
    
    n_wm: int = 500
    n_neg: int = 1000
    
    secret_salt: str = "seal_watermark_salt_2024"
    
    transforms: List[str] = field(default_factory=lambda: [
        "Original", "Rotation_75", "JPEG_25", "Crop_Scale_0.75",
        "Gaussian_Blur_8x8", "Gaussian_Noise_0.1", "Brightness_U06"
    ])
    
    shuffle_seed: int = 42
    global_seed: int = 0
