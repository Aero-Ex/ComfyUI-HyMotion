<p align="center">
  <h1 align="center">ComfyUI-HyMotion</h1>
  <p align="center"><strong>Text-to-Motion Generation · Robust FBX Retargeting · Interactive 3D Preview</strong></p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  <a href="https://github.com/comfyanonymous/ComfyUI"><img alt="ComfyUI" src="https://img.shields.io/badge/ComfyUI-compatible-green.svg"></a>
</p>

A full-featured ComfyUI implementation of **HY-MOTION 1.0**, enabling high-fidelity human motion generation from text prompts. This node pack provides a complete pipeline from text description to rigged FBX animation, with real-time 3D preview capabilities and robust retargeting to custom skeletons.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎬 Text-to-Motion** | Generate realistic human animations from natural language descriptions using DiT (Diffusion Transformer) architecture |
| **🔄 Robust Retargeting** | Transfer SMPL-H motions to custom skeletons (Mixamo, UE5, UniRig) with intelligent fuzzy mapping, canonical T-pose support, and geometric fallbacks |
| **🖼️ Interactive 3D Viewer** | Real-time Three.js preview with transform gizmos (G/R/S keys), resizable viewport, and smooth sub-frame interpolation |
| **📝 Prompt Enhancement** | AI-powered prompt rewriting with automatic duration estimation using Text2MotionPrompter |
| **💾 Multiple Export Formats** | Export to FBX (with skeleton & textures) or NPZ (raw SMPL-H data) |
| **🎮 SMPL Integration** | Convert motion capture data (GVHMR/MotionCapture output) to HY-Motion format for retargeting |
| **🏎️ MHR GPU Converter** | Ultra-fast GPU-accelerated MHR-to-HyMotion conversion with finger fitting and auto-alignment |
| **⚡ GGUF Support** | Memory-efficient text encoding with quantized Qwen3 models via ComfyUI-GGUF |

---

## 🚀 Installation

### Option 1: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Aero-Ex/ComfyUI-HyMotion
cd ComfyUI-HyMotion
pip install -r requirements.txt
```

> [!NOTE]
> The installation includes `fbxsdkpy` from a custom PyPI index for FBX support.

---

## 📦 Model Downloads

### Core Motion Models

Download and place in `ComfyUI/models/hymotion/`:

| Model | Size | Description | Download |
|-------|------|-------------|----------|
| **HY-Motion-1.0** | ~2GB | Full quality model | [📥 latest.ckpt](https://huggingface.co/Aero-Ex/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0/latest.ckpt) |
| **HY-Motion-1.0-Lite** | ~800MB | Faster, lower VRAM | [📥 latest.ckpt](https://huggingface.co/Aero-Ex/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0-Lite/latest.ckpt) |

**Directory structure:**
```
ComfyUI/models/hymotion/
├── HY-Motion-1.0/
│   └── latest.ckpt
└── HY-Motion-1.0-Lite/
    └── latest.ckpt
```

### Text Encoders

Download and place in `ComfyUI/models/text_encoders/`:

| Encoder | Format | Description | Download |
|---------|--------|-------------|----------|
| **CLIP ViT-L/14** | SafeTensors | Visual-text encoder | [📥 clip-vit-large-patch14.safetensors](https://huggingface.co/Aero-Ex/Hy-Motion1.0/resolve/main/text_encoders/clip-vit-large-patch14.safetensors) |
| **Qwen3-8B** | FP8 | Language model (16GB+ VRAM) | [📥 Qwen3-8B_fp8.safetensors](https://huggingface.co/Aero-Ex/Hy-Motion1.0/resolve/main/text_encoders/Qwen3-8B_fp8.safetensors) |
| **Qwen3-8B** | GGUF | Quantized (lower VRAM) | [📁 Browse GGUF options](https://huggingface.co/Aero-Ex/Hy-Motion1.0/tree/main/text_encoders/Qwen3-8B-GGUF) |

---

## 🔧 Node Reference

### Core Pipeline Nodes (`HY-Motion/modular`)

| Node | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| **HY-Motion DiT Loader** | Load the motion generation model | Model name, device | `HYMOTION_DIT` |
| **HY-Motion Text Encoder Loader** | Load CLIP + Qwen3 text encoders | CLIP model, LLM model, device | `HYMOTION_TEXT_ENCODER` |
| **HY-Motion Text Encode** | Convert text prompt to embeddings | Text encoder, prompt | `HYMOTION_TEXT_EMBEDS` |
| **HY-Motion Sampler** | Generate motion from embeddings | DiT, embeds, duration, seed, CFG | `HYMOTION_DATA` |
| **HY-Motion Export FBX** | Export motion to FBX file | Motion data, template, FPS, scale | FBX file path |
| **HY-Motion Prompt Rewrite** | Enhance prompt & estimate duration | Raw prompt, enhancer mode | Rewritten prompt, duration |

### Preview Nodes (`HY-Motion/view`)

| Node | Purpose | Description |
|------|---------|-------------|
| **HY-Motion 2D Motion Preview** | Render motion as image sequence | Matplotlib-based skeleton visualization with camera controls |
| **HY-Motion 3D Model Loader** | Load FBX/GLB/GLTF/OBJ files | Interactive Three.js viewer with transform controls |
| **HY-Motion FBX Player** | Play FBX animations (legacy) | Dedicated FBX playback from output directory |

### Utility Nodes (`HY-Motion/utils`)

| Node | Purpose |
|------|---------|
| **HY-Motion Save NPZ** | Export raw SMPL-H motion data to NumPy format |
| **HY-Motion Retarget to FBX** | Transfer SMPL-H motion to custom skeletons with robust bone mapping |
| **HY-Motion SMPL to Data** | Convert SMPL parameters from motion capture to HY-Motion format |

### MHR Conversion Nodes (`HY-Motion/loaders`, `HY-Motion/converters`)

| Node | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| **HY-Motion MHR Loader** | Load MHR vertex data from NPZ files | MHR NPZ name | `MHR_DATA` |
| **HY-Motion MHR Converter (GPU)** | Ultra-fast GPU-accelerated conversion with opt-in finger fitting | MHR Data, fit_hands, flip_orientation | `HYMOTION_DATA` |

---

## 📋 Workflows

Sample workflows are included in the `workflows/` directory:
- `workflows/HunyuanMotion.json`: Basic text-to-motion pipeline.
- `workflows/Text_Video-To-3DMotion.json`: Advanced pipeline with video-to-motion and retargeting.

---

## ⚙️ Technical Details

### Robust Retargeting
The retargeting engine has been significantly improved to support a wide range of skeletons:
- **UE5 Support**: Dedicated mapping for Unreal Engine 5 Mannequin skeletons.
- **UniRig/ArticulationXL**: Built-in support for UniRig-detected skeletons.
- **Canonical T-Pose**: Automatic rest-pose normalization for stable retargeting from NPZ sources.
- **Geometric Fallbacks**: Intelligent bone matching based on relative hierarchy and position when name matching fails.
- **In-Place Support**: Option to lock horizontal movement for game-ready animations.

### Requirements
- Python 3.11+
- CUDA-compatible GPU (5GB+ VRAM recommended)
- ComfyUI (latest version)

---

## 📂 Repository Structure
```
ComfyUI-HyMotion/
├── hymotion/           # Core logic and utilities
│   ├── network/        # DiT and Text Encoder architectures
│   ├── pipeline/       # Diffusion sampling logic
│   └── utils/          # Retargeting, loaders, and math utilities
├── nodes_modular.py    # Primary ComfyUI node definitions
├── nodes_2d_preview.py # 2D visualization nodes
├── nodes_3d_viewer.py # 3D viewer integration
├── web/                # Frontend for 3D viewer
├── workflows/          # Example ComfyUI workflows
└── tests/              # Parity and integration tests
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

## 📜 Credits

This project is based on the following research and implementations:

- [HY-Motion](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) - High-fidelity human motion generation.
- [MHR (Mesh-based Human Reconstruction)](https://github.com/facebookresearch/MHR) - High-fidelity mesh reconstruction from video (the basis for our GPU conversion pipeline).
- [SMPL-X](https://github.com/vchoutas/smplx) - Expressive body model.

**ComfyUI implementation by [Aero-Ex](https://github.com/Aero-Ex).**
