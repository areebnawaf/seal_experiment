import sys
import json
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.seal import SEAL
from src.detector import Detector
from src.models import load_models

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe, blip_model, blip_proc, embedder, inv_scheduler = load_models(
        config, device, dtype
    )
    seal = SEAL(pipe, blip_model, blip_proc, embedder, config, device, dtype)
    detector = Detector(seal, inv_scheduler)
    
    results_path = Path("results/results.csv")
    results = []
    
    for t in config.transforms:
        wm_dir = Path(f"results/wm/{t}")
        if not wm_dir.exists():
            continue
        
        for img_path in tqdm(sorted(wm_dir.glob("wm_*.png")), desc=f"WM-{t}"):
            idx = int(img_path.stem.split('_')[1])
            
            img = Image.open(img_path).convert("RGB")
            score, dists, cap, _ = detector.detect(img, details=True)
            
            results.append({
                'idx': idx,
                'label': 1,
                'transform': t,
                'm': score,
                'caption': cap
            })
            
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv(results_path, index=False)
    
    neg_dir = Path("results/neg/Original")
    for img_path in tqdm(sorted(neg_dir.glob("neg_*.png")), desc="NEG"):
        idx = int(img_path.stem.split('_')[1])
        
        img = Image.open(img_path).convert("RGB")
        score, dists, cap, _ = detector.detect(img, details=True)
        
        results.append({
            'idx': idx,
            'label': 0,
            'transform': 'Original',
            'm': score,
            'caption': cap
        })
        
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(results_path, index=False)
    
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"✓ Saved {len(results)} detection results")

if __name__ == "__main__":
    main()
