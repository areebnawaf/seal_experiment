import hashlib
import numpy as np
import torch
from typing import List

class SEAL:
    
    def __init__(self, pipe, captioner, cap_proc, embedder, config, device, dtype):
        self.pipe = pipe
        self.captioner = captioner
        self.cap_proc = cap_proc
        self.embedder = embedder
        self.config = config
        self.device = device
        self.dtype = dtype
        
        self.latent_h = self.latent_w = 64
        self.latent_c = 4
        self.embed_dim = embedder.get_sentence_embedding_dimension()
    
    def _hash(self, *args):
        h = hashlib.sha256()
        for a in args:
            h.update(str(a).encode())
        h.update(self.config.secret_salt.encode())
        return int(h.hexdigest()[:8], 16)
    
    def _proj_vec(self, i, j):
        rng = np.random.RandomState(self._hash(i, j))
        return rng.randn(self.embed_dim).astype(np.float32)
    
    def _simhash(self, v, i):
        bits = ''.join(
            '1' if np.dot(v, self._proj_vec(i, j)) >= 0 else '0'
            for j in range(self.config.bits_per_patch)
        )
        return self._hash(bits, i)
    
    def _patch_noise(self, seed, size):
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(size, generator=gen)
    
    def caption(self, img):
        inputs = self.cap_proc(images=img, return_tensors="pt").to(
            self.device, dtype=self.dtype
        )
        with torch.no_grad():
            ids = self.captioner.generate(**inputs, max_new_tokens=50)
        return self.cap_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    
    def caption_variations(self, img, n_variations):
        inputs = self.cap_proc(images=img, return_tensors="pt").to(
            self.device, dtype=self.dtype
        )
        captions = set()
        
        with torch.no_grad():
            ids = self.captioner.generate(**inputs, max_new_tokens=50)
            captions.add(
                self.cap_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            )
            
            n_beams = min(n_variations, 10)
            ids = self.captioner.generate(
                **inputs, max_new_tokens=50, num_beams=n_beams, 
                num_return_sequences=n_beams
            )
            for c in self.cap_proc.batch_decode(ids, skip_special_tokens=True):
                captions.add(c.strip())
            
            remaining = n_variations - len(captions)
            if remaining > 0:
                ids = self.captioner.generate(
                    **inputs, max_new_tokens=50, do_sample=True, 
                    top_p=0.9, temperature=1.0, num_return_sequences=remaining
                )
                for c in self.cap_proc.batch_decode(ids, skip_special_tokens=True):
                    captions.add(c.strip())
        
        return list(captions)
    
    def embed(self, caption):
        with torch.no_grad():
            return self.embedder.encode(
                caption, convert_to_numpy=True, normalize_embeddings=True
            )
    
    def embed_batch(self, captions):
        with torch.no_grad():
            return self.embedder.encode(
                captions, convert_to_numpy=True, normalize_embeddings=True
            )
