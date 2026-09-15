"""
Multi-Task TransUNet Package.
"""
from .multitask_transunet import MultiTaskTransUNet
from .encoder import ResNetEncoder
from .transformer import TransformerBottleneck
from .decoder import UNetDecoder
from .segmentation_head import SegmentationHead
from .depth_head import DepthHead
from .uncertainty_head import UncertaintyHead

TransUNet = MultiTaskTransUNet

def build_transunet(cfg=None):
    return MultiTaskTransUNet()

__all__ = [
    "MultiTaskTransUNet",
    "TransUNet",
    "build_transunet",
    "ResNetEncoder",
    "TransformerBottleneck",
    "UNetDecoder",
    "SegmentationHead",
    "DepthHead",
    "UncertaintyHead",
]
