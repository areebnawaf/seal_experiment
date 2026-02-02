"""Unit tests for SEAL watermarking"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import Config, SEAL


class TestSEAL:
    """Test cases for SEAL watermarker"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config(
            n_patches=64,  # Smaller for testing
            bits_per_patch=7,
            patch_size=2,
            tau=2.3,
            n_wm=10,
            n_neg=20
        )
    
    @pytest.fixture
    def mock_models(self, config):
        """Create mock models"""
        # Mock Stable Diffusion pipeline
        pipe = Mock()
        pipe.vae = Mock()
        pipe.unet = Mock()
        pipe.tokenizer = Mock()
        pipe.text_encoder = Mock()
        
        # Mock BLIP-2
        captioner = Mock()
        cap_proc = Mock()
        
        # Mock embedder
        embedder = Mock()
        embedder.get_sentence_embedding_dimension.return_value = 768
        embedder.encode = Mock(return_value=np.random.randn(768).astype(np.float32))
        
        device = torch.device('cpu')
        dtype = torch.float32
        
        return pipe, captioner, cap_proc, embedder, device, dtype
    
    def test_seal_initialization(self, config, mock_models):
        """Test SEAL initializes correctly"""
        seal = SEAL(*mock_models, config)
        
        assert seal.config == config
        assert seal.latent_h == 64
        assert seal.latent_w == 64
        assert seal.latent_c == 4
        assert seal.embed_dim == 768
    
    def test_hash_function(self, config, mock_models):
        """Test cryptographic hash function"""
        seal = SEAL(*mock_models, config)
        
        # Same input should give same hash
        hash1 = seal._hash("test", "input")
        hash2 = seal._hash("test", "input")
        assert hash1 == hash2
        
        # Different input should give different hash
        hash3 = seal._hash("different", "input")
        assert hash1 != hash3
        
        # Hash should be 32-bit integer
        assert 0 <= hash1 < 2**32
    
    def test_proj_vec(self, config, mock_models):
        """Test projection vector generation"""
        seal = SEAL(*mock_models, config)
        
        # Same indices should give same vector
        vec1 = seal._proj_vec(0, 0)
        vec2 = seal._proj_vec(0, 0)
        np.testing.assert_array_equal(vec1, vec2)
        
        # Different indices should give different vectors
        vec3 = seal._proj_vec(0, 1)
        assert not np.array_equal(vec1, vec3)
        
        # Vector should have correct dimension
        assert vec1.shape == (768,)
        assert vec1.dtype == np.float32
    
    def test_simhash(self, config, mock_models):
        """Test SimHash encoding"""
        seal = SEAL(*mock_models, config)
        
        v = np.random.randn(768).astype(np.float32)
        
        # Same vector and patch should give same hash
        hash1 = seal._simhash(v, 0)
        hash2 = seal._simhash(v, 0)
        assert hash1 == hash2
        
        # Different patch should give different hash
        hash3 = seal._simhash(v, 1)
        assert hash1 != hash3
        
        # Hash should be 32-bit integer
        assert 0 <= hash1 < 2**32
    
    def test_patch_noise(self, config, mock_models):
        """Test deterministic noise generation"""
        seal = SEAL(*mock_models, config)
        
        seed = 12345
        size = (1, 4, 2, 2)
        
        # Same seed should give same noise
        noise1 = seal._patch_noise(seed, size)
        noise2 = seal._patch_noise(seed, size)
        torch.testing.assert_close(noise1, noise2)
        
        # Different seed should give different noise
        noise3 = seal._patch_noise(54321, size)
        assert not torch.allclose(noise1, noise3)
        
        # Noise should have correct shape
        assert noise1.shape == size
    
    def test_make_wm_noise(self, config, mock_models):
        """Test watermark noise generation"""
        seal = SEAL(*mock_models, config)
        
        v = np.random.randn(768).astype(np.float32)
        noise = seal.make_wm_noise(v)
        
        # Noise should have correct shape
        assert noise.shape == (1, 4, 64, 64)
        
        # Same vector should give same noise
        noise2 = seal.make_wm_noise(v)
        torch.testing.assert_close(noise, noise2)
        
        # Different vector should give different noise
        v2 = np.random.randn(768).astype(np.float32)
        noise3 = seal.make_wm_noise(v2)
        assert not torch.allclose(noise, noise3)
    
    def test_embed_batch(self, config, mock_models):
        """Test batch embedding"""
        pipe, captioner, cap_proc, embedder, device, dtype = mock_models
        
        # Mock batch embedding
        captions = ["caption 1", "caption 2", "caption 3"]
        embeddings = np.random.randn(len(captions), 768).astype(np.float32)
        embedder.encode = Mock(return_value=embeddings)
        
        seal = SEAL(pipe, captioner, cap_proc, embedder, config, device, dtype)
        result = seal.embed_batch(captions)
        
        # Should call embedder once with all captions
        embedder.encode.assert_called_once()
        
        # Result should have correct shape
        assert result.shape == (3, 768)


class TestSimHashProperties:
    """Test mathematical properties of SimHash"""
    
    def test_simhash_similarity(self):
        """Test that similar vectors have similar hashes"""
        config = Config()
        mock_pipe = Mock()
        mock_captioner = Mock()
        mock_cap_proc = Mock()
        mock_embedder = Mock()
        mock_embedder.get_sentence_embedding_dimension.return_value = 768
        
        seal = SEAL(
            mock_pipe, mock_captioner, mock_cap_proc, mock_embedder,
            config, torch.device('cpu'), torch.float32
        )
        
        # Create similar vectors (small angle)
        v1 = np.random.randn(768).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)
        
        # v2 is v1 with small noise
        noise = np.random.randn(768).astype(np.float32) * 0.01
        v2 = v1 + noise
        v2 = v2 / np.linalg.norm(v2)
        
        # Count matching bits across multiple patches
        matches = 0
        total_bits = 0
        for i in range(100):
            for j in range(config.bits_per_patch):
                bit1 = 1 if np.dot(v1, seal._proj_vec(i, j)) >= 0 else 0
                bit2 = 1 if np.dot(v2, seal._proj_vec(i, j)) >= 0 else 0
                if bit1 == bit2:
                    matches += 1
                total_bits += 1
        
        # Similar vectors should have high bit agreement (>80%)
        agreement = matches / total_bits
        assert agreement > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
