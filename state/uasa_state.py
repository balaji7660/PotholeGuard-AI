"""
state/uasa_state.py
UASA State Generator with strict 117-dimensional vector validation.
"""
import numpy as np
from state.uasa import UASAStateGenerator, STATE_DIM

UASA_STATE_DIM: int = STATE_DIM

def generate_uasa_state(
    pred_mask,
    depth_map,
    uncertainty_map,
    vehicle_state=None,
    generator=None
):
    """
    Generate and validate the exact 117-dimensional UASA state vector.
    """
    if generator is None:
        generator = UASAStateGenerator()
    
    seg_float = (pred_mask > 0).astype(np.float32) if pred_mask.dtype != np.float32 else pred_mask
    depth_float = depth_map.astype(np.float32)
    unc_float = uncertainty_map.astype(np.float32)

    if hasattr(generator, "compute"):
        state_vector = generator.compute(
            segmentation=seg_float,
            depth=depth_float,
            uncertainty=unc_float
        )
    elif hasattr(generator, "generate"):
        uasa_obj = generator.generate(
            pred_mask=pred_mask,
            depth_map=depth_map,
            uncertainty_map=uncertainty_map,
            vehicle_state=vehicle_state
        )
        state_vector = uasa_obj.vector if hasattr(uasa_obj, "vector") else uasa_obj
    else:
        state_vector = generator(pred_mask=pred_mask, depth_map=depth_map, uncertainty_map=uncertainty_map)

    assert len(state_vector) == 117, f"UASA state vector dimension mismatch: expected 117, got {len(state_vector)}"
    return state_vector

__all__ = ["UASAStateGenerator", "generate_uasa_state", "UASA_STATE_DIM", "STATE_DIM"]
