"""SEAL: Semantic Aware Image Watermarking Implementation"""

import hashlib
import numpy as np
import torch
from typing import Tuple
from PIL import Image


class SEAL:
    """
    SEAL: Semantic Aware Image Watermarking
    
    Implements the SEAL watermarking method from:
    "SEAL: Semantic Aware Image Watermarking" (ICCV 2024)
    
    Paper: https://arxiv.org/abs/your-paper-id
    """
    
    def __init__(self, pipe, captioner, cap_proc, embedder, config, device, dtype):
        """
        Initialize SEAL watermarker
        
        Args:
            pipe: Stable Diffusion pipeline
            captioner: BLIP-2 model for captioning
            cap_proc: BLIP-2 processor
            embedder: Sentence transformer for embeddings
            config: Configuration object
            device: torch device
            dtype: torch dtype
        """
        self.pipe = pipe
        self.captioner = captioner
        self.cap_proc = cap_proc
        self.embedder = embedder
        self.config = config
        self.device = device
        self.dtype = dtype
        
        # Latent space dimensions for Stable Diffusion
        self.latent_h = self.latent_w = 64
        self.latent_c = 4
        self.embed_dim = embedder.get_sentence_embedding_dimension()
    
    def _hash(self, *args) -> int:
        """
        Cryptographic hash function with secret salt
        
        Args:
            *args: Arguments to hash
            
        Returns:
            32-bit integer hash
        """
        h = hashlib.sha256()
        for a in args:
            h.update(str(a).encode())
        h.update(self.config.secret_salt.encode())
        return int(h.hexdigest()[:8], 16)
    
    def _proj_vec(self, i: int, j: int) -> np.ndarray:
        """
        Generate random projection vector for SimHash
        
        Args:
            i: Patch index
            j: Bit index
            
        Returns:
            Random projection vector
        """
        rng = np.random.RandomState(self._hash(i, j))
        return rng.randn(self.embed_dim).astype(np.float32)
    
    def _simhash(self, v: np.ndarray, i: int) -> int:
        """
        SimHash: Convert semantic vector to patch seed
        
        Args:
            v: Semantic embedding vector
            i: Patch index
            
        Returns:
            Seed for patch noise generation
        """
        bits = ''.join(
            '1' if np.dot(v, self._proj_vec(i, j)) >= 0 else '0'
            for j in range(self.config.bits_per_patch)
        )
        return self._hash(bits, i)
    
    def _patch_noise(self, seed: int, size: Tuple) -> torch.Tensor:
        """
        Generate deterministic noise patch from seed
        
        Args:
            seed: Random seed
            size: Tensor size (batch, channels, height, width)
            
        Returns:
            Gaussian noise tensor
        """
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(size, generator=gen)
    
    def caption(self, img: Image.Image) -> str:
        """
        Caption image using BLIP-2 (greedy decoding)
        
        Args:
            img: PIL Image
            
        Returns:
            Caption string
        """
        inputs = self.cap_proc(
            images=img, 
            return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            ids = self.captioner.generate(**inputs, max_new_tokens=50)
        
        return self.cap_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    
    def caption_variations(self, img: Image.Image, n_variations: int) -> list:
        """
        Generate multiple caption variations via beam/sample decoding
        
        This is critical for per-patch search detection (Algorithm 3).
        
        Args:
            img: PIL Image
            n_variations: Number of caption variations to generate
            
        Returns:
            List of caption strings
        """
        inputs = self.cap_proc(
            images=img,
            return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        
        captions = set()
        
        with torch.no_grad():
            # 1) Greedy decode
            ids = self.captioner.generate(**inputs, max_new_tokens=50)
            greedy = self.cap_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            captions.add(greedy)
            
            # 2) Beam search for diverse captions
            n_beams = min(n_variations, 10)
            ids = self.captioner.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=n_beams,
                num_return_sequences=n_beams
            )
            beam_caps = self.cap_proc.batch_decode(ids, skip_special_tokens=True)
            for c in beam_caps:
                captions.add(c.strip())
            
            # 3) Nucleus sampling for additional diversity
            remaining = n_variations - len(captions)
            if remaining > 0:
                ids = self.captioner.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    num_return_sequences=remaining
                )
                sample_caps = self.cap_proc.batch_decode(ids, skip_special_tokens=True)
                for c in sample_caps:
                    captions.add(c.strip())
        
        return list(captions)
    
    def embed(self, caption: str) -> np.ndarray:
        """
        Embed caption to semantic vector
        
        Args:
            caption: Text caption
            
        Returns:
            Normalized embedding vector
        """
        with torch.no_grad():
            return self.embedder.encode(
                caption,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
    
    def embed_batch(self, captions: list) -> np.ndarray:
        """
        Embed multiple captions in batch
        
        Args:
            captions: List of text captions
            
        Returns:
            Array of normalized embedding vectors
        """
        with torch.no_grad():
            return self.embedder.encode(
                captions,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
    
    def make_wm_noise(self, v: np.ndarray) -> torch.Tensor:
        """
        Build full watermark noise lattice from semantic embedding
        
        Args:
            v: Semantic embedding vector
            
        Returns:
            Watermarked noise tensor (1, 4, 64, 64)
        """
        ps = self.config.patch_size
        ppr = self.latent_w // ps  # Patches per row
        noise = torch.zeros(1, self.latent_c, self.latent_h, self.latent_w)
        n_patches = min(self.config.n_patches, ppr * (self.latent_h // ps))
        
        for i in range(n_patches):
            r, c = (i // ppr) * ps, (i % ppr) * ps
            if r + ps <= self.latent_h and c + ps <= self.latent_w:
                seed = self._simhash(v, i)
                noise[:, :, r:r+ps, c:c+ps] = self._patch_noise(
                    seed, (1, self.latent_c, ps, ps)
                )
        
        return noise
    
    def gen_wm(self, prompt: str, seed: int = None) -> Tuple:
        """
        Generate watermarked image (Algorithm 1 from paper)
        
        Steps:
        1. Generate proxy image to estimate semantics
        2. Caption proxy and get semantic embedding
        3. Generate watermarked noise from embedding
        4. Generate final image with watermarked noise
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed (optional)
            
        Returns:
            (image, caption, embedding, watermark_noise)
        """
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed else None
        
        # Step 1 & 2: Proxy generation and captioning
        with torch.no_grad():
            proxy = self.pipe(
                prompt,
                num_inference_steps=self.config.gen_steps,
                guidance_scale=self.config.guidance_scale,
                generator=gen
            ).images[0]
        
        cap = self.caption(proxy)
        v = self.embed(cap)
        
        # Step 3 & 4: Generate with watermark noise
        wm_noise = self.make_wm_noise(v).to(self.device, self.dtype)
        with torch.no_grad():
            final = self.pipe(
                prompt,
                latents=wm_noise,
                num_inference_steps=self.config.gen_steps,
                guidance_scale=self.config.guidance_scale
            ).images[0]
        
        return final, cap, v, wm_noise
    
    def gen_clean(self, prompt: str, seed: int = None) -> Image.Image:
        """
        Generate clean (non-watermarked) image
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed (optional)
            
        Returns:
            PIL Image
        """
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed else None
        
        with torch.no_grad():
            return self.pipe(
                prompt,
                num_inference_steps=self.config.gen_steps,
                guidance_scale=self.config.guidance_scale,
                generator=gen
            ).images[0]
