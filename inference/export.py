"""
Model Export Utilities for PotholeGuard-AI.
Exports MultiTaskTransUNet to ONNX and TorchScript formats for edge / mobile acceleration.
"""
import torch
import os
from models.transunet import MultiTaskTransUNet

def export_model(checkpoint_path: str = None, output_dir: str = "exported_models"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    model = MultiTaskTransUNet(pretrained_encoder=False).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    # 1. TorchScript Export
    ts_path = os.path.join(output_dir, "transunet_traced.pt")
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(ts_path)
        print(f"[+] Exported TorchScript model to: {ts_path}")
    except Exception as e:
        print(f"[!] TorchScript export error: {e}")

    # 2. ONNX Export
    onnx_path = os.path.join(output_dir, "transunet.onnx")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input_rgb"],
            output_names=["seg_logits", "depth_map", "uncertainty_map"],
            opset_version=14,
            dynamic_axes={"input_rgb": {0: "batch_size"}}
        )
        print(f"[+] Exported ONNX model to: {onnx_path}")
    except Exception as e:
        print(f"[!] ONNX export error: {e}")

if __name__ == "__main__":
    export_model()
