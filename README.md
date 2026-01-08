<p align="center">
  <h1 align="center">ComfyUI-HyMotion</h1>
  <p align="center"><strong>Text-to-Motion Generation · FBX Retargeting · Interactive 3D Preview</strong></p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  <a href="https://github.com/comfyanonymous/ComfyUI"><img alt="ComfyUI" src="https://img.shields.io/badge/ComfyUI-compatible-green.svg"></a>
</p>

A full-featured ComfyUI implementation of **HY-MOTION 1.0**, enabling high-fidelity human motion generation from text prompts. This node pack provides a complete pipeline from text description to rigged FBX animation, with real-time 3D preview capabilities.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎬 Text-to-Motion** | Generate realistic human animations from natural language descriptions using DiT (Diffusion Transformer) architecture |
| **🔄 FBX Retargeting** | Transfer generated SMPL-H motions to  custom skeleton (Mixamo) with intelligent fuzzy bone mapping |
| **🖼️ Interactive 3D Viewer** | Real-time Three.js preview with transform gizmos (G/R/S keys), resizable viewport, and smooth sub-frame interpolation |
| **📝 Prompt Enhancement** | AI-powered prompt rewriting with automatic duration estimation using Text2MotionPrompter |
| **💾 Multiple Export Formats** | Export to FBX (with skeleton) or NPZ (raw SMPL-H data) |
| **🎮 SMPL Integration** | Convert motion capture data (GVHMR/MotionCapture output) to HY-Motion format for retargeting |
| **⚡ GGUF Support** | Memory-efficient text encoding with quantized Qwen3 models via ComfyUI-GGUF |

---

## 🚀 Installation

### Option 1: ComfyUI Manager (Recommended)
Search for **"ComfyUI-HyMotion"** in the ComfyUI Manager and click Install.

### Option 2: Manual Installation
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
| **HY-Motion-1.0** | ~2GB | Full quality model | [📥 latest.ckpt](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0/latest.ckpt) |
| **HY-Motion-1.0-Lite** | ~800MB | Faster, lower VRAM | [📥 latest.ckpt](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0-Lite/latest.ckpt) |

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
| **CLIP ViT-L/14** | SafeTensors | Visual-text encoder | [📥 clip-vit-large-patch14.safetensors](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/text_encoders/clip-vit-large-patch14.safetensors) |
| **Qwen3-8B** | FP8 | Language model (16GB+ VRAM) | [📥 Qwen3-8B_fp8.safetensors](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/text_encoders/Qwen3-8B_fp8.safetensors) |
| **Qwen3-8B** | GGUF | Quantized (lower VRAM) | [📁 Browse GGUF options](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/tree/main/text_encoders/Qwen3-8B-GGUF) |

> [!IMPORTANT]
> To use GGUF text encoders, install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) first.

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
| **HY-Motion Retarget to FBX** | Transfer SMPL-H motion to custom skeletons with bone mapping |
| **HY-Motion SMPL to Data** | Convert SMPL parameters from motion capture to HY-Motion format |

---

## 🎮 3D Viewer Controls

The integrated Three.js viewer supports interactive manipulation:

| Key | Action |
|-----|--------|
| **G** | Translate mode (move object) |
| **R** | Rotate mode |
| **S** | Scale mode |
| **Mouse drag** | Orbit camera |
| **Scroll** | Zoom in/out |
| **Right-click drag** | Pan camera |

The viewport is resizable and supports sub-frame interpolation for smooth playback regardless of source FPS.

---

## 📋 Workflows

A sample workflow is included at `workflows/HunyuanMotion.json`. Load it via **File → Load** in ComfyUI.

### Basic Text-to-FBX Pipeline

```
[Text Prompt] → [Prompt Rewrite] → [Text Encode] → [Sampler] → [Export FBX] → [3D Viewer]
                      ↑                  ↑              ↑
               [Text Encoder]     [DiT Loader]    [Duration]
```

### Retargeting Pipeline

```
[Motion Data] → [Retarget to FBX] → [3D Viewer]
                      ↑
              [Custom Skeleton FBX]
```

---

## ⚙️ Technical Details

### Architecture
- **Backbone**: Multimodal DiT (Diffusion Transformer) with 18 layers
- **Text Encoders**: CLIP ViT-L/14 (768D) + Qwen3-8B (4096D)
- **Motion Representation**: SMPL-H format (52 joints, 156 body parameters)
- **Sampling**: ODE-based diffusion with configurable solvers (dopri5, euler, etc.)

### Supported Formats
| Format | Import | Export | Description |
|--------|:------:|:------:|-------------|
| FBX | ✅ | ✅ | Industry-standard animation format |

### Requirements
- Python 3.11+
- CUDA-compatible GPU (5GB+ VRAM recommended)
- ComfyUI (latest version)

### Dependencies
```
fbxsdkpy==2020.1.post2    # FBX SDK bindings
torchdiffeq>=0.2.5        # ODE solvers
transforms3d>=0.4.2       # 3D transformations
omegaconf>=2.3.0          # Configuration management
accelerate>=0.30.0        # Model optimization
huggingface_hub>=0.30.0   # Model downloads
bitsandbytes>=0.42.0      # Quantization support
safetensors>=0.4.0        # Model loading
scipy>=1.10.0             # Signal processing
opencv-python>=4.8.0      # Image processing
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

## 📜 Credits

This project is based on the [HyMotion](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) research.  
**ComfyUI implementation by [Aero-Ex](https://github.com/Aero-Ex).**

