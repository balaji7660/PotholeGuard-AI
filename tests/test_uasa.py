"""
Unit tests for 117-dimensional UASA state validation.
"""
import numpy as np
import pytest
from state.uasa_state import generate_uasa_state, UASAStateGenerator

def test_uasa_dimension_assertion():
    gen = UASAStateGenerator()
    dummy_mask = np.zeros((512, 512), dtype=np.uint8)
    dummy_mask[200:300, 200:300] = 1
    dummy_depth = np.ones((512, 512), dtype=np.float32) * 0.5
    dummy_unc = np.ones((512, 512), dtype=np.float32) * 0.1
    
    state = generate_uasa_state(
        pred_mask=dummy_mask,
        depth_map=dummy_depth,
        uncertainty_map=dummy_unc,
        generator=gen
    )
    
    assert len(state) == 117
    assert isinstance(state, np.ndarray)
    assert not np.isnan(state).any()
    assert not np.isinf(state).any()
