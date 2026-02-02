import sys
import os
import json
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.seal import SEAL
from src.models import load_models

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    output_dir = Path("results/wm/Original")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipe, blip_model, blip_proc, embedder, inv_scheduler = load_models(
        config, device, dtype
    )
    
    seal = SEAL(pipe, blip_model, blip_proc, embedder, config, device, dtype)
    
    prompts = load_prompts(config.n_wm)  
    
    print(f"\nGenerating {config.n_wm} watermarked images...")
    
    for idx, prompt in enumerate(tqdm(prompts)):
        img_path = output_dir / f"wm_{idx:05d}.png"
        json_path = output_dir / f"wm_{idx:05d}.json"
        
        if img_path.exists():
            continue
        
        z_pre = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        x_pre = pipe(
            prompt=prompt,
            latents=z_pre,
            num_inference_steps=config.gen_steps,
            guidance_scale=config.guidance_scale
        ).images[0]
        
        caption = seal.caption(x_pre)
        v = seal.embed(caption)
        
        z = torch.zeros(1, 4, 64, 64, device=device, dtype=dtype)
        ps = config.patch_size
        for i in range(config.n_patches):
            r, c = (i // 32) * ps, (i % 32) * ps
            seed = seal._simhash(v, i)
            z[:, :, r:r+ps, c:c+ps] = seal._patch_noise(
                seed, (1, 4, ps, ps)
            ).to(device, dtype=dtype)
        
        img = pipe(
            prompt=prompt,
            latents=z,
            num_inference_steps=config.gen_steps,
            guidance_scale=config.guidance_scale
        ).images[0]
        
        img.save(img_path)
        with open(json_path, 'w') as f:
            json.dump({
                'idx': idx,
                'prompt': prompt,
                'caption': caption
            }, f)
    
    print(f"✓ Generated {config.n_wm} watermarked images")

if __name__ == "__main__":
    main()
