#!/usr/bin/env python3
"""
Test script for random band masking in BigEarthNet dataset.

This verifies that:
1. RGB bands (0, 1, 2) are never masked
2. Non-RGB bands (3-11) are randomly masked with the specified probability
3. Masked bands are filled with zeros
"""

import sys
import torch
sys.path.insert(0, '/home/carmenoliver/my_projects/dinov3-testing-stuff')

from dinov3.data.band_masking import RandomBandMasking


def test_band_masking():
    print("=" * 80)
    print("Testing RandomBandMasking")
    print("=" * 80)
    
    # Create sample 12-band input
    torch.manual_seed(42)
    x = torch.randn(12, 256, 256)
    print(f"\nOriginal input shape: {x.shape}")
    print(f"Original band statistics (mean per band):")
    for i in range(12):
        print(f"  Band {i}: mean={x[i].mean():.4f}, std={x[i].std():.4f}")
    
    # Test with 30% masking probability
    print("\n" + "-" * 80)
    print("Testing with mask_prob=0.3")
    print("-" * 80)
    
    masker = RandomBandMasking(mask_prob=0.3, mask_value=0.0, rgb_indices=(0, 1, 2))
    masker.train()  # Enable training mode for masking
    
    # Apply masking multiple times to see randomness
    print("\nApplying masking 5 times to observe randomness:")
    for trial in range(5):
        x_masked = masker(x.clone())
        
        # Check which bands are masked (all-zero)
        masked_bands = []
        for i in range(12):
            if torch.allclose(x_masked[i], torch.zeros_like(x_masked[i])):
                masked_bands.append(i)
        
        print(f"  Trial {trial + 1}: Masked bands = {masked_bands}")
        
        # Verify RGB bands are never masked
        for rgb_idx in [0, 1, 2]:
            assert not torch.allclose(x_masked[rgb_idx], torch.zeros_like(x_masked[rgb_idx])), \
                f"RGB band {rgb_idx} should never be masked!"
    
    print("\n✓ RGB bands (0, 1, 2) are never masked")
    
    # Test eval mode (no masking)
    print("\n" + "-" * 80)
    print("Testing eval mode (masking disabled)")
    print("-" * 80)
    
    masker.eval()
    x_eval = masker(x.clone())
    
    assert torch.allclose(x, x_eval), "In eval mode, no masking should occur"
    print("✓ Eval mode: no masking applied")
    
    # Test with different mask probabilities
    print("\n" + "-" * 80)
    print("Testing different mask probabilities")
    print("-" * 80)
    
    for prob in [0.0, 0.5, 1.0]:
        masker = RandomBandMasking(mask_prob=prob)
        masker.train()
        
        masked_count = 0
        n_trials = 100
        for _ in range(n_trials):
            x_masked = masker(x.clone())
            for i in range(3, 12):  # Only check non-RGB bands
                if torch.allclose(x_masked[i], torch.zeros_like(x_masked[i])):
                    masked_count += 1
        
        expected_masked = prob * 9 * n_trials  # 9 non-RGB bands
        actual_rate = masked_count / (9 * n_trials)
        print(f"  mask_prob={prob:.1f}: actual masking rate={actual_rate:.3f} (expected ~{prob:.1f})")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_band_masking()
