"""
Unit tests for Multi-Task TransUNet architecture.
"""
import torch
import pytest
from models.transunet import MultiTaskTransUNet

def test_transunet_forward_pass():
    model = MultiTaskTransUNet(pretrained_encoder=False, embed_dim=256, transformer_layers=2, transformer_heads=4)
    model.eval()
    
    # Input tensor [B=2, C=3, H=256, W=256]
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        outputs = model(dummy_input, mc_dropout=True)
        
    assert "seg" in outputs
    assert "depth" in outputs
    assert "uncertainty" in outputs
    
    assert outputs["seg"].shape == (2, 1, 256, 256)
    assert outputs["depth"].shape == (2, 1, 256, 256)
    assert outputs["uncertainty"].shape == (2, 1, 256, 256)
    
    # Check bounds
    assert torch.all(outputs["depth"] >= 0.0) and torch.all(outputs["depth"] <= 1.0)
    assert torch.all(outputs["uncertainty"] >= 0.0) and torch.all(outputs["uncertainty"] <= 1.0)
