import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer

def load_models(config, device, dtype, hf_token=None):
    
    print("[1/3] Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.sd_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        token=hf_token
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    print(f"✓ Loaded: {config.sd_model}")
    
    print("\n[2/3] Loading BLIP-2...")
    blip_proc = Blip2Processor.from_pretrained(
        config.captioner_model, token=hf_token
    )
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        config.captioner_model,
        torch_dtype=dtype,
        token=hf_token
    ).to(device).eval()
    print(f"✓ Loaded: {config.captioner_model}")
    
    print("\n[3/3] Loading Embedder...")
    embedder = SentenceTransformer(config.embedder_model).to(device)
    print(f"✓ Loaded: {config.embedder_model}")
    
    inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    
    print("\n✓ All models loaded")
    return pipe, blip_model, blip_proc, embedder, inv_scheduler
