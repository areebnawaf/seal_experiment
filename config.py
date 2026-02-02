"""Configuration for SEAL experiment"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """SEAL experiment configuration"""
    
    # Model IDs
    # NOTE: Using ModelScope for SD 2.1 as HuggingFace version is gated/restricted
    # This uses the EXACT setup that produced the verified results
    sd_model: str = "AI-ModelScope/stable-diffusion-2-1-base"  # ModelScope (works immediately)
    sd_model_hf: str = "stabilityai/stable-diffusion-2-1"  # HF fallback (requires token)
    captioner_model: str = "Salesforce/blip2-flan-t5-xl"
    embedder_model: str = "kasraarabi/finetuned-caption-embedding"
    fallback_embedder: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    
    # Model source preference
    use_modelscope: bool = True  # Set to False to try HuggingFace (requires access)
    modelscope_cache_dir: str = "./models_cache"  # Where to cache ModelScope downloads
    
    # SEAL parameters (from paper)
    n_patches: int = 1024      # K: Number of patches
    bits_per_patch: int = 7    # B: Bits per SimHash
    patch_size: int = 2        # Patch size in latent space
    tau: float = 2.3           # Distance threshold (Supplement §9.1)
    
    # Generation parameters
    gen_steps: int = 50        # Diffusion steps for generation
    inv_steps: int = 50        # Inversion steps
    guidance_scale: float = 7.5
    
    # Security
    secret_salt: str = "seal_watermark_salt_2024"
    
    # Per-patch search parameters (Paper §4.2, Algorithm 3)
    n_search_captions: int = 16  # Caption variations for detection
    
    # Dataset parameters
    n_wm: int = 500            # Number of watermarked images
    n_neg: int = 1000          # Number of negative images
    prompt_dataset: str = "Gustavosta/Stable-Diffusion-Prompts"
    
    # Seeds for reproducibility
    shuffle_seed: int = 42     # Seed for shuffling prompts
    global_seed: int = 0       # Base seed for generation
    
    # Transforms (Paper §4, Supplement §9.3)
    transforms: List[str] = field(default_factory=lambda: [
        "Original",
        "Rotation_75",
        "JPEG_25",
        "Crop_Scale_0.75",
        "Gaussian_Blur_8x8",
        "Gaussian_Noise_0.1",
        "Brightness_U06"
    ])
    
    # Detection thresholds
    detection_threshold: int = 12  # Minimum patches to match (Paper: nmatch)
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.n_patches > 0, "n_patches must be positive"
        assert self.bits_per_patch > 0, "bits_per_patch must be positive"
        assert self.patch_size > 0, "patch_size must be positive"
        assert self.tau > 0, "tau must be positive"
        assert 0 < self.guidance_scale <= 20, "guidance_scale must be in (0, 20]"
        assert self.n_wm > 0, "n_wm must be positive"
        assert self.n_neg > 0, "n_neg must be positive"
        assert self.n_search_captions > 0, "n_search_captions must be positive"


def load_config(config_path: str = None) -> Config:
    """Load configuration from YAML file or use defaults"""
    if config_path is None:
        return Config()
    
    import yaml
    from pathlib import Path
    
    path = Path(config_path)
    if not path.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return Config()
    
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)
