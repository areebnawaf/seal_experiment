"""SEAL Watermark Detector with Per-Patch Search"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional


class Detector:
    """
    SEAL Watermark Detector with per-patch search
    
    Implements Algorithm 3 from the paper (§4.2):
    - Generate multiple caption variations
    - For each patch, search over all caption embeddings
    - Find minimum distance to expected noise
    - Count matches below threshold
    """
    
    def __init__(self, seal, inv_scheduler):
        """
        Initialize detector
        
        Args:
            seal: SEAL watermarker instance
            inv_scheduler: DDIM inverse scheduler
        """
        self.seal = seal
        self.inv_sched = inv_scheduler
        self.config = seal.config
    
    def invert(self, img: Image.Image, prompt: str = "") -> torch.Tensor:
        """
        DDIM inversion to recover initial noise latent
        
        Args:
            img: PIL Image to invert
            prompt: Text prompt (usually empty for watermark detection)
            
        Returns:
            Inverted noise tensor (1, 4, 64, 64)
        """
        # Resize if needed
        if img.size != (512, 512):
            img = img.resize((512, 512))
        
        with torch.no_grad():
            # Encode image to latent space
            img_tensor = torch.from_numpy(
                np.array(img).astype(np.float32) / 127.5 - 1
            ).permute(2, 0, 1).unsqueeze(0).to(self.seal.device, self.seal.dtype)
            
            latent = self.seal.pipe.vae.encode(img_tensor).latent_dist.mean
            latent = latent * self.seal.pipe.vae.config.scaling_factor
            
            # Get text embeddings
            text_input = self.seal.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.seal.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = self.seal.pipe.text_encoder(
                text_input.input_ids.to(self.seal.device)
            )[0]
            
            # DDIM inversion loop
            self.inv_sched.set_timesteps(self.config.inv_steps)
            for timestep in self.inv_sched.timesteps:
                # Predict noise
                noise_pred = self.seal.pipe.unet(
                    latent,
                    timestep,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Inverse step
                latent = self.inv_sched.step(
                    noise_pred,
                    timestep,
                    latent
                ).prev_sample
        
        return latent
    
    def _patch_region(self, i: int) -> Tuple[int, int]:
        """
        Get (row, col) top-left corner for patch index i
        
        Args:
            i: Patch index
            
        Returns:
            (row, col) coordinates
        """
        ps = self.config.patch_size
        patches_per_row = self.seal.latent_w // ps
        row = (i // patches_per_row) * ps
        col = (i % patches_per_row) * ps
        return row, col
    
    def _extract_patch(self, latent: torch.Tensor, i: int) -> torch.Tensor:
        """
        Extract patch at index i from latent
        
        Args:
            latent: Latent tensor (1, 4, 64, 64)
            i: Patch index
            
        Returns:
            Patch tensor (4, patch_size, patch_size)
        """
        ps = self.config.patch_size
        row, col = self._patch_region(i)
        return latent[0, :, row:row+ps, col:col+ps].float()
    
    def detect(
        self,
        img: Image.Image,
        details: bool = False
    ) -> Tuple:
        """
        Per-patch search detection (Algorithm 3)
        
        This is the key detection algorithm from the paper that:
        1. Generates N caption variations
        2. For each patch, searches over all N embeddings
        3. Counts patches with distance below threshold τ
        
        Args:
            img: PIL Image to detect watermark in
            details: If True, return additional information
            
        Returns:
            If details=False: m (number of matching patches)
            If details=True: (m, distances, caption, embeddings)
        """
        ps = self.config.patch_size
        n_patches = min(
            self.config.n_patches,
            (self.seal.latent_w // ps) * (self.seal.latent_h // ps)
        )
        
        # Step 1: Generate caption variations (greedy + beam + sampling)
        captions = self.seal.caption_variations(img, self.config.n_search_captions)
        
        # Step 2: Batch embed all captions
        embeddings = self.seal.embed_batch(captions)
        
        # Step 3: DDIM inversion to recover initial noise
        inverted_noise = self.invert(img)
        
        # Step 4: Per-patch search
        best_distances = np.full(n_patches, float('inf'))
        
        for i in range(n_patches):
            row, col = self._patch_region(i)
            
            # Skip patches outside latent bounds
            if row + ps > self.seal.latent_h or col + ps > self.seal.latent_w:
                continue
            
            # Extract inverted patch (move to CPU for comparison)
            inverted_patch = self._extract_patch(inverted_noise, i).cpu()
            
            # Search over all caption embeddings
            for embedding_j in embeddings:
                # Generate expected patch from this embedding
                seed_j = self.seal._simhash(embedding_j, i)
                expected_patch = self.seal._patch_noise(
                    seed_j,
                    (1, self.seal.latent_c, ps, ps)
                )[0].float()
                
                # Compute L2 distance
                distance_j = torch.norm(inverted_patch - expected_patch).item()
                
                # Keep minimum distance
                if distance_j < best_distances[i]:
                    best_distances[i] = distance_j
        
        # Step 5: Count matches below threshold τ
        m = int(np.sum(best_distances < self.config.tau))
        
        if details:
            # Return greedy caption for logging
            greedy_caption = captions[0] if captions else ""
            return m, best_distances, greedy_caption, embeddings
        
        return m
    
    def is_watermarked(self, img: Image.Image) -> bool:
        """
        Simple watermark detection decision
        
        Args:
            img: PIL Image to check
            
        Returns:
            True if watermarked, False otherwise
        """
        m = self.detect(img, details=False)
        return m >= self.config.detection_threshold
    
    def detect_tampering(self, img: Image.Image, threshold_percentile: int = 80) -> dict:
        """
        Detect localized tampering using patch distance map
        
        This enables detecting attacks like the "Cat Attack" where
        an object is pasted into a watermarked image.
        
        Args:
            img: PIL Image to analyze
            threshold_percentile: Percentile for high-distance threshold
            
        Returns:
            Dictionary with tampering analysis
        """
        m, distances, caption, embeddings = self.detect(img, details=True)
        
        # Threshold at high percentile
        valid_distances = distances[distances < np.inf]
        if len(valid_distances) == 0:
            return {
                'watermarked': False,
                'tampered': False,
                'high_distance_patches': 0
            }
        
        threshold = np.percentile(valid_distances, threshold_percentile)
        high_dist_patches = np.sum(distances > threshold)
        
        # Spatial clustering analysis (simplified)
        # In full implementation, use connected components analysis
        is_tampered = high_dist_patches > (len(valid_distances) * 0.1)
        
        return {
            'watermarked': m >= self.config.detection_threshold,
            'tampered': is_tampered,
            'high_distance_patches': int(high_dist_patches),
            'mean_distance': float(np.mean(valid_distances)),
            'distance_map': distances
        }
