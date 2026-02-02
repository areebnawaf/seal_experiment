import numpy as np
import torch
from PIL import Image

class Detector:
    
    def __init__(self, seal, inv_scheduler):
        self.seal = seal
        self.inv_sched = inv_scheduler
        self.config = seal.config
    
    def invert(self, img, prompt=""):
        if img.size != (512, 512):
            img = img.resize((512, 512))
        
        with torch.no_grad():
            t = torch.from_numpy(
                np.array(img).astype(np.float32) / 127.5 - 1
            ).permute(2, 0, 1).unsqueeze(0).to(self.seal.device, self.seal.dtype)
            
            lat = self.seal.pipe.vae.encode(t).latent_dist.mean
            lat = lat * self.seal.pipe.vae.config.scaling_factor
            
            txt_in = self.seal.pipe.tokenizer(
                prompt, padding="max_length",
                max_length=self.seal.pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            txt_emb = self.seal.pipe.text_encoder(
                txt_in.input_ids.to(self.seal.device)
            )[0]
            
            self.inv_sched.set_timesteps(self.config.inv_steps)
            for ts in self.inv_sched.timesteps:
                noise = self.seal.pipe.unet(
                    lat, ts, encoder_hidden_states=txt_emb
                ).sample
                lat = self.inv_sched.step(noise, ts, lat).prev_sample
        
        return lat
    
    def _patch_region(self, i):
        """Get patch coordinates"""
        ps = self.config.patch_size
        ppr = self.seal.latent_w // ps
        r = (i // ppr) * ps
        c = (i % ppr) * ps
        return r, c
    
    def _extract_patch(self, latent, i):
        """Extract patch from latent"""
        ps = self.config.patch_size
        r, c = self._patch_region(i)
        return latent[0, :, r:r+ps, c:c+ps].float()
    
    def detect(self, img, details=False):
        """Per-patch search detection"""
        ps = self.config.patch_size
        n_patches = min(
            self.config.n_patches,
            (self.seal.latent_w // ps) * (self.seal.latent_h // ps)
        )
        
        captions = self.seal.caption_variations(img, self.config.n_search_captions)
        
        embeddings = self.seal.embed_batch(captions)
        
        inv = self.invert(img)
        
        best_dists = np.full(n_patches, float('inf'))
        
        for i in range(n_patches):
            r, c = self._patch_region(i)
            if r + ps > self.seal.latent_h or c + ps > self.seal.latent_w:
                continue
            
            inv_patch = self._extract_patch(inv, i).cpu()
            
            for v_j in embeddings:
                seed_j = self.seal._simhash(v_j, i)
                exp_patch = self.seal._patch_noise(
                    seed_j, (1, self.seal.latent_c, ps, ps)
                )[0].float()
                
                dist_j = torch.norm(inv_patch - exp_patch).item()
                if dist_j < best_dists[i]:
                    best_dists[i] = dist_j
        
        m = int(np.sum(best_dists < self.config.tau))
        
        if details:
            greedy_caption = captions[0] if captions else ""
            return m, best_dists, greedy_caption, embeddings
        return m
