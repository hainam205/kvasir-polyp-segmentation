from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.ducknet import DuckNet
from models.segnet import SegNet
from models.unet import UNet
from utils.train import load_checkpoint


# ── Cấu hình model ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: type[torch.nn.Module]
    checkpoint: Path
    image_size: int
    threshold: float = 0.5
    threshold_file: Path | None = None
    model_kwargs: dict[str, Any] | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "UNet": ModelSpec(
        name="UNet",
        builder=UNet,
        checkpoint=PROJECT_ROOT / "checkpoints" / "unet_fair_last.pth",
        image_size=256,
        model_kwargs={"in_channels": 3, "num_classes": 1},
    ),
    "SegNet": ModelSpec(
        name="SegNet",
        builder=SegNet,
        checkpoint=PROJECT_ROOT / "checkpoints" / "segnet_fair_last.pth",
        image_size=256,
        model_kwargs={"in_channels": 3, "num_classes": 1},
    ),
    "DuckNet": ModelSpec(
        name="DuckNet",
        builder=DuckNet,
        checkpoint=PROJECT_ROOT / "checkpoints" / "ducknet_fair_last.pth",
        image_size=256,
        model_kwargs={"in_channels": 3, "num_classes": 1, "base_channels": 32, "dropout": 0.1},
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_available(name: str) -> bool:
    return MODEL_SPECS[name].checkpoint.is_file()


def get_threshold(name: str) -> float:
    spec = MODEL_SPECS[name]
    if spec.threshold_file and spec.threshold_file.is_file():
        try:
            v = float(spec.threshold_file.read_text(encoding="utf-8").strip())
            return float(np.clip(v, 0.05, 0.95))
        except (OSError, ValueError):
            pass
    return spec.threshold


@st.cache_resource(show_spinner=False)
def load_model(name: str) -> torch.nn.Module:
    spec = MODEL_SPECS[name]
    model = spec.builder(**(spec.model_kwargs or {"in_channels": 3, "num_classes": 1}))
    load_checkpoint(model, spec.checkpoint, map_location="cpu")
    return model.eval()


def preprocess(image_rgb: np.ndarray, size: int) -> tuple[torch.Tensor, np.ndarray]:
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std, resized


@torch.no_grad()
def run_model(image_rgb: np.ndarray, name: str, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = MODEL_SPECS[name]
    tensor, resized = preprocess(image_rgb, spec.image_size)
    prob = torch.sigmoid(load_model(name)(tensor))[0, 0].cpu().numpy()
    mask = (prob >= threshold).astype(np.uint8) * 255
    prob_vis = (prob * 255).clip(0, 255).astype(np.uint8)
    return resized, prob_vis, mask


def overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colored = np.zeros_like(image_rgb)
    colored[..., 1] = mask
    return cv2.addWeighted(image_rgb, 0.7, colored, 0.3, 0.0)


def to_png(image: np.ndarray) -> bytes:
    img = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img)
    return io.BytesIO(buf.tobytes()).getvalue()


def decode_upload(f) -> np.ndarray:
    arr = np.frombuffer(f.read(), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không đọc được ảnh.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── UI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Polyp Segmentation",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Phân đoạn Polyp Đại tràng")
    st.caption("Tải ảnh nội soi lên, chọn model và xem kết quả phân đoạn theo thời gian thực.")

    available = [n for n in MODEL_SPECS if is_available(n)]
    missing   = [n for n in MODEL_SPECS if not is_available(n)]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Cấu hình")

        if missing:
            st.warning(f"Chưa có checkpoint: {', '.join(missing)}")

        mode = st.radio(
            "Chế độ",
            options=["Một model", "So sánh 3 model"],
            index=0,
            help="'So sánh 3 model' chạy cả 3 cùng lúc trên cùng 1 ảnh"
        )

        if mode == "Một model":
            selected = st.selectbox(
                "Chọn model",
                options=list(MODEL_SPECS.keys()),
                index=list(MODEL_SPECS.keys()).index("DuckNet"),
            )
            auto_thr = get_threshold(selected)
            threshold = st.slider(
                "Ngưỡng phân loại",
                min_value=0.05, max_value=0.95,
                value=float(round(auto_thr / 0.05) * 0.05),
                step=0.05,
            )
            st.caption(f"Ngưỡng gợi ý: {auto_thr:.3f}")
        else:
            threshold = st.slider(
                "Ngưỡng phân loại (dùng chung)",
                min_value=0.05, max_value=0.95,
                value=0.50, step=0.05,
            )

        st.divider()
        uploaded = st.file_uploader(
            "📂 Tải ảnh lên",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        )

        st.divider()
        st.markdown("**Checkpoint hiện có:**")
        for name in MODEL_SPECS:
            icon = "✅" if is_available(name) else "❌"
            st.markdown(f"{icon} {name}")

    # ── Main area ─────────────────────────────────────────────────────────────
    if uploaded is None:
        st.info("👈 Tải ảnh lên từ thanh bên trái để bắt đầu.")
        return

    image_rgb = decode_upload(uploaded)

    # ── Chế độ 1 model ────────────────────────────────────────────────────────
    if mode == "Một model":
        if not is_available(selected):
            st.error(f"Model **{selected}** chưa có checkpoint.")
            return

        with st.spinner(f"Đang chạy {selected}..."):
            resized, prob_vis, mask = run_model(image_rgb, selected, threshold)
        ov = overlay(resized, mask)
        ratio = mask.mean() / 255.0

        st.subheader(f"Kết quả — {selected}")
        c1, c2, c3 = st.columns(3)
        c1.image(resized,   caption="Ảnh đầu vào",    use_container_width=True)
        c2.image(mask,      caption="Mask nhị phân",   use_container_width=True, clamp=True)
        c3.image(ov,        caption="Overlay",         use_container_width=True)

        with st.expander("🗺️ Bản đồ xác suất", expanded=False):
            st.image(prob_vis, caption="Xác suất từng pixel (trắng = cao)", use_container_width=True, clamp=True)

        st.metric("Tỉ lệ pixel polyp dự đoán", f"{ratio:.2%}")

        col_dl1, col_dl2 = st.columns(2)
        stem = Path(uploaded.name).stem
        col_dl1.download_button("⬇️ Tải mask PNG",    to_png(mask), f"{stem}_{selected.lower()}_mask.png",    "image/png")
        col_dl2.download_button("⬇️ Tải overlay PNG", to_png(ov),   f"{stem}_{selected.lower()}_overlay.png", "image/png")

    # ── Chế độ so sánh 3 model ────────────────────────────────────────────────
    else:
        if not available:
            st.error("Chưa có checkpoint nào.")
            return

        st.subheader("So sánh 3 model — cùng ảnh đầu vào")

        # Hàng tiêu đề: ảnh gốc + tên 3 model
        cols = st.columns(4)
        cols[0].markdown("**Ảnh gốc**")
        for i, name in enumerate(MODEL_SPECS):
            icon = "✅" if is_available(name) else "❌"
            cols[i + 1].markdown(f"**{icon} {name}**")

        # Hàng mask
        row_mask = st.columns(4)
        # Hàng overlay
        row_over = st.columns(4)
        # Hàng metric
        row_stat = st.columns(4)

        img_display = cv2.resize(image_rgb, (256, 256))
        row_mask[0].image(img_display, caption="Ảnh đầu vào", use_container_width=True)
        row_over[0].image(img_display, caption="Ảnh đầu vào", use_container_width=True)
        row_stat[0].markdown("")

        for i, name in enumerate(MODEL_SPECS):
            col_idx = i + 1
            if not is_available(name):
                row_mask[col_idx].warning("Chưa có checkpoint")
                row_over[col_idx].warning("Chưa có checkpoint")
                continue

            with st.spinner(f"Đang chạy {name}..."):
                resized, prob_vis, mask = run_model(image_rgb, name, threshold)
            ov = overlay(resized, mask)
            ratio = mask.mean() / 255.0

            row_mask[col_idx].image(mask, caption="Mask",    use_container_width=True, clamp=True)
            row_over[col_idx].image(ov,   caption="Overlay", use_container_width=True)
            row_stat[col_idx].metric(f"Polyp ({name})", f"{ratio:.2%}")


if __name__ == "__main__":
    main()
