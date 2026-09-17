"""Gradio interactive demo for robust digit recognition."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from corruptions import CORRUPTION_FN
from evaluate import get_device, load_model

CKPT = ROOT / "outputs" / "fasternet_aug.pt"
DEVICE = get_device()
MODEL = None


def _ensure_model():
    global MODEL
    if MODEL is None:
        if not CKPT.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {CKPT}. Train first: python train.py --model fasternet --aug"
            )
        MODEL, _ = load_model(CKPT, DEVICE)
    return MODEL


def _to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("L").resize((28, 28))
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr)[None, None, ...].to(DEVICE)


@torch.no_grad()
def predict(image, corruption: str, reject_threshold: float, use_tta: bool):
    if image is None:
        return "Please draw or upload a digit.", None
    model = _ensure_model()
    x = _to_tensor(image)
    x = CORRUPTION_FN[corruption](x)
    if use_tta:
        from evaluate import predict_with_rotation_tta

        probs = predict_with_rotation_tta(model, x)[0]
    else:
        probs = F.softmax(model(x), dim=1)[0]
    conf, pred = probs.max(0)
    conf = float(conf)
    pred = int(pred)
    ranking = sorted([(i, float(probs[i])) for i in range(10)], key=lambda t: -t[1])[:5]
    rank_txt = ", ".join([f"{i}:{p:.3f}" for i, p in ranking])
    if conf < reject_threshold:
        msg = f"REJECT (not sure)\nbest guess={pred}  p={conf:.3f}\nTop5: {rank_txt}"
    else:
        msg = f"Prediction={pred}  confidence={conf:.3f}\nTop5: {rank_txt}"
    # return preview of corrupted tensor
    preview = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
    return msg, Image.fromarray(preview)


def main():
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", image_mode="L", label="Draw / upload digit"),
            gr.Dropdown(choices=list(CORRUPTION_FN.keys()), value="clean", label="Corruption"),
            gr.Slider(0.5, 0.99, value=0.7, step=0.01, label="Reject threshold"),
            gr.Checkbox(value=False, label="Rotation TTA"),
        ],
        outputs=[
            gr.Textbox(label="Result"),
            gr.Image(label="Model input (after corruption)"),
        ],
        title="CISC3024 — Robust Digit Recognition (FasterNet-mini)",
        description="AI Assignment #1 demo. CHE CHI HIN, Angus (UC325182).",
    )
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
