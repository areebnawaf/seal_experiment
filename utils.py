"""Utility Functions for SEAL Experiment"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer


def load_models(config, device, dtype, hf_token=None):
    """
    Load all required models for SEAL
    
    NOTE: This uses ModelScope's snapshot_download for SD 2.1 by default since 
    the HuggingFace version is gated/restricted. This is the EXACT method that
    produced the verified results.
    
    Args:
        config: Configuration object
        device: torch device
        dtype: torch dtype
        hf_token: HuggingFace token (optional, required for HF SD 2.1)
        
    Returns:
        (pipe, blip_model, blip_proc, embedder, inv_scheduler)
    """
    print("[1/3] Loading Stable Diffusion 2.1...")
    
    # Try ModelScope first (exact working method)
    if config.use_modelscope:
        print(f"  Attempting ModelScope snapshot download...")
        try:
            # Import modelscope
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except ImportError:
                print("  Installing modelscope...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "-q"])
                from modelscope.hub.snapshot_download import snapshot_download
            
            # Download model to cache
            print(f"  Downloading: {config.sd_model}")
            model_dir = snapshot_download(
                config.sd_model,
                cache_dir=config.modelscope_cache_dir
            )
            print(f"  ✓ Downloaded to: {model_dir}")
            
            # Load from local cache
            pipe = StableDiffusionPipeline.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True
            )
            print(f"  ✓ Loaded from ModelScope: {config.sd_model}")
            
        except Exception as e:
            print(f"  ⚠️ ModelScope failed: {e}")
            print(f"  Falling back to HuggingFace: {config.sd_model_hf}")
            pipe = StableDiffusionPipeline.from_pretrained(
                config.sd_model_hf,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                token=hf_token
            )
            print(f"  ✓ Loaded from HuggingFace: {config.sd_model_hf}")
    else:
        # Try HuggingFace directly (requires token)
        print(f"  Attempting HuggingFace: {config.sd_model_hf}")
        pipe = StableDiffusionPipeline.from_pretrained(
            config.sd_model_hf,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            token=hf_token
        )
        print(f"  ✓ Loaded from HuggingFace: {config.sd_model_hf}")
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    print("  ✓ SD 2.1 ready")
    
    print("\n[2/3] Loading BLIP-2...")
    blip_proc = Blip2Processor.from_pretrained(
        config.captioner_model,
        token=hf_token
    )
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        config.captioner_model,
        torch_dtype=dtype,
        token=hf_token
    ).to(device).eval()
    print(f"✓ Loaded: {config.captioner_model}")
    
    print("\n[3/3] Loading Embedder...")
    try:
        embedder = SentenceTransformer(config.embedder_model).to(device)
        print(f"✓ Loaded: {config.embedder_model}")
    except Exception as e:
        print(f"⚠️  Could not load {config.embedder_model}: {e}")
        print(f"Falling back to {config.fallback_embedder}")
        embedder = SentenceTransformer(config.fallback_embedder).to(device)
        print(f"✓ Loaded: {config.fallback_embedder}")
    
    # Inverse scheduler for detection
    inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    
    print("\n✓ All models loaded")
    return pipe, blip_model, blip_proc, embedder, inv_scheduler


def save_run_metadata(output_dir: Path, config, additional_info: Dict = None):
    """
    Save experiment metadata to JSON file
    
    Args:
        output_dir: Output directory path
        config: Configuration object
        additional_info: Additional metadata to save
    """
    metadata = {
        "run_id": output_dir.name,
        "timestamp": datetime.now().isoformat(),
        "detection_mode": "per_patch_search",
        "config": {
            "n_wm": config.n_wm,
            "n_neg": config.n_neg,
            "n_patches": config.n_patches,
            "bits_per_patch": config.bits_per_patch,
            "tau": config.tau,
            "n_search_captions": config.n_search_captions,
            "transforms": config.transforms,
        },
        "models": {
            "stable_diffusion": config.sd_model,
            "captioner": config.captioner_model,
            "embedder": config.embedder_model,
        }
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {output_dir / 'run_meta.json'}")


def load_prompts_from_dataset(dataset_name: str, n_prompts: int, shuffle_seed: int = 42):
    """
    Load prompts from HuggingFace dataset
    
    Args:
        dataset_name: Dataset name (e.g., "Gustavosta/Stable-Diffusion-Prompts")
        n_prompts: Number of prompts to load
        shuffle_seed: Random seed for shuffling
        
    Returns:
        List of prompt strings
    """
    from datasets import load_dataset
    import random
    
    print(f"Loading prompts from {dataset_name}...")
    
    try:
        ds = load_dataset(dataset_name, split="train")
        
        # Extract prompts (handle different dataset formats)
        if "Prompt" in ds.column_names:
            all_prompts = ds["Prompt"]
        elif "prompt" in ds.column_names:
            all_prompts = ds["prompt"]
        elif "text" in ds.column_names:
            all_prompts = ds["text"]
        else:
            raise ValueError(f"Could not find prompt column in dataset. Columns: {ds.column_names}")
        
        # Shuffle and select
        random.seed(shuffle_seed)
        all_prompts = list(all_prompts)
        random.shuffle(all_prompts)
        
        selected = all_prompts[:n_prompts]
        print(f"✓ Loaded {len(selected)} prompts")
        
        return selected
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Falling back to sample prompts...")
        return generate_sample_prompts(n_prompts)


def generate_sample_prompts(n_prompts: int) -> list:
    """
    Generate sample prompts as fallback
    
    Args:
        n_prompts: Number of prompts to generate
        
    Returns:
        List of prompt strings
    """
    templates = [
        "A photo of {}",
        "An oil painting of {}",
        "A digital art of {}",
        "A beautiful {} at sunset",
        "A detailed illustration of {}",
    ]
    
    subjects = [
        "a beach", "a mountain", "a forest", "a city", "a cat",
        "a dog", "a bird", "a flower", "a tree", "a car",
        "a house", "a castle", "a spaceship", "a robot", "a dragon"
    ]
    
    prompts = []
    for i in range(n_prompts):
        template = templates[i % len(templates)]
        subject = subjects[(i // len(templates)) % len(subjects)]
        prompts.append(template.format(subject))
    
    return prompts


def setup_output_directory(base_dir: str = "results") -> Path:
    """
    Create timestamped output directory
    
    Args:
        base_dir: Base directory for results
        
    Returns:
        Path to output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"seal_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "wm").mkdir(exist_ok=True)
    (output_dir / "neg").mkdir(exist_ok=True)
    
    print(f"✓ Output directory: {output_dir}")
    return output_dir


def get_device_info() -> Dict[str, Any]:
    """
    Get device and memory information
    
    Returns:
        Dictionary with device info
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    info = {
        "device": str(device),
        "dtype": str(dtype),
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["n_gpus"] = torch.cuda.device_count()
    
    return info
