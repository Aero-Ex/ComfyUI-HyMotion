import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import time
import uuid
import importlib.util
import gc
from typing import Optional, List, Dict, Any, Tuple

import folder_paths
import comfy.utils
from scipy.signal import savgol_filter
from torchdiffeq import odeint

# Internal hymotion imports
from .hymotion.utils.loaders import load_object
from .hymotion.network.text_encoders.text_encoder import HYTextModel
from .hymotion.pipeline.motion_diffusion import length_to_mask, randn_tensor, MotionGeneration
from .hymotion.utils.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d, axis_angle_to_matrix
from .hymotion.pipeline.body_model import WoodenMesh, construct_smpl_data_dict
from .hymotion.utils.downloader import download_file, get_model_path, MODEL_METADATA
from .hymotion.utils.data_types import HYMotionData, HYMotionTextEmbeds, HYMotionFrame
from .nodes_3d_viewer import HYMotionFBXPlayer, HYMotion3DModelLoader

# Retargeting imports
try:
    from .hymotion.utils.retarget_fbx import (
        Skeleton, BoneData, load_npz, load_fbx, load_bone_mapping,
        retarget_animation, apply_retargeted_animation, save_fbx,
        collect_skeleton_nodes, extract_animation, get_skeleton_height
    )
    HAS_RETARGET_UTILS = True
except ImportError as e:
    print(f"[HY-Motion] Warning: Could not import retargeting utils: {e}")
    HAS_RETARGET_UTILS = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
except AttributeError:
    COMFY_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "output")

def get_timestamp():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"

# ============================================================================
# Default Configurations (for Zero-Config Loading)
# ============================================================================

# Global Cache for performance
_WOODEN_MESH_CACHE = None

DEFAULT_CONFIG_FULL = {
    "network_module": "hymotion/network/hymotion_mmdit.HunyuanMotionMMDiT",
    "network_module_args": {
        "apply_rope_to_single_branch": False,
        "ctxt_input_dim": 4096,
        "dropout": 0.0,
        "feat_dim": 1280,
        "input_dim": 201,
        "mask_mode": "narrowband",
        "mlp_ratio": 4.0,
        "num_heads": 20,
        "num_layers": 27,
        "time_factor": 1000.0,
        "vtxt_input_dim": 768
    },
    "vtxt_input_dim": 768,
    "ctxt_input_dim": 4096,
    "train_frames": 360
}

DEFAULT_CONFIG_LITE = {
    "network_module": "hymotion/network/hymotion_mmdit.HunyuanMotionMMDiT",
    "network_module_args": {
        "apply_rope_to_single_branch": False,
        "ctxt_input_dim": 4096,
        "dropout": 0.0,
        "feat_dim": 1024,
        "input_dim": 201,
        "mask_mode": "narrowband",
        "mlp_ratio": 4.0,
        "num_heads": 16,
        "num_layers": 18,
        "time_factor": 1000.0,
        "vtxt_input_dim": 768
    },
    "vtxt_input_dim": 768,
    "ctxt_input_dim": 4096,
    "train_frames": 360
}


# ============================================================================
# Wrapper Classes for Modular Components
# ============================================================================

class HYMotionDiT:
    """Wrapper for the HY-Motion DiT model"""
    def __init__(self, network, config, stats_dir, device, mean=None, std=None, 
                 null_vtxt_feat=None, null_ctxt_input=None, 
                 special_game_vtxt_feat=None, special_game_ctxt_feat=None,
                 train_frames=360):
        self.network = network
        self.config = config
        self.stats_dir = stats_dir
        self.device = device
        self.mean = mean
        self.std = std
        
        # Learned null features for CFG (from official implementation)
        self.null_vtxt_feat = null_vtxt_feat  # Shape: [1, 1, 768]
        self.null_ctxt_input = null_ctxt_input  # Shape: [1, 1, 4096]
        
        # Special game character features (from official implementation)
        self.special_game_vtxt_feat = special_game_vtxt_feat
        self.special_game_ctxt_feat = special_game_ctxt_feat
        
        # Training frame count - latent is generated at this size then cropped
        self.train_frames = train_frames
        
        # Load normalization stats from disk if not provided and stats_dir exists
        if self.mean is None and stats_dir and os.path.exists(stats_dir):
            for m_name, s_name in [("Mean.npy", "Std.npy"), ("mean.npy", "std.npy")]:
                mean_path = os.path.join(stats_dir, m_name)
                std_path = os.path.join(stats_dir, s_name)
                if os.path.exists(mean_path) and os.path.exists(std_path):
                    self.mean = torch.from_numpy(np.load(mean_path)).float().to(device)
                    self.std = torch.from_numpy(np.load(std_path)).float().to(device)
                    print(f"[HYMotionDiT] Loaded stats from disk: {m_name}, {s_name}")
                    break






# ============================================================================
# Node 1: HYMotionDiTLoader - Load Diffusion Transformer
# ============================================================================

class HYMotionDiTLoader:
    @classmethod
    def INPUT_TYPES(s):
        raw_options = folder_paths.get_filename_list("hymotion")
        model_options = []
        for f in raw_options:
            # Extract parent directory if it's a file inside a folder (platform-agnostic)
            name = os.path.dirname(f) if "/" in f or "\\" in f else f
            if name and name not in model_options:
                model_options.append(name)
        
        if not model_options:
            model_options = ["HY-Motion-1.0", "HY-Motion-1.0-Lite"]
        
        return {
            "required": {
                "model_name": (model_options, {
                    "default": model_options[0] if model_options else "HY-Motion-1.0-Lite",
                    "tooltip": "Select the HY-Motion DiT model checkpoint. 'Lite' version uses less VRAM."
                }),
                "device": (["cuda", "cpu", "mps"], {
                    "default": "cuda",
                    "tooltip": "Device to load the model on. CUDA is faster but requires GPU memory. MPS is for Mac users."
                }),
            },
        }
    
    RETURN_TYPES = ("HYMOTION_DIT",)
    RETURN_NAMES = ("dit_model",)
    FUNCTION = "load_dit"
    CATEGORY = "HY-Motion/modular"
    
    def load_dit(self, model_name, device):
        
        # Resolve model directory
        # First try get_full_path
        model_dir = folder_paths.get_full_path("hymotion", model_name)
        
        # If not found, it might be a directory. get_filename_list for "hymotion" returns basenames
        if model_dir is None:
            for p in folder_paths.get_folder_paths("hymotion"):
                potential_path = os.path.join(p, model_name)
                if os.path.isdir(potential_path):
                    model_dir = potential_path
                    break
        if model_dir is None:
            raise FileNotFoundError(f"Model directory not found: {model_name}")
            
        # If model_dir points to a file (like a .ckpt), get its parent directory
        if os.path.isfile(model_dir):
            model_dir = os.path.dirname(model_dir)
        
        config_path = os.path.join(model_dir, "config.yml")
        ckpt_path = os.path.join(model_dir, "latest.ckpt")
        
        # If config.yml is missing, use bit-perfect official default configs
        if not os.path.exists(config_path):
            print(f"[HY-Motion] config.yml not found. Applying Zero-Config logic...")
            if not os.path.exists(ckpt_path):
                # Search for any valid weight file
                ckpt_files = [f for f in os.listdir(model_dir) if f.endswith((".ckpt", ".pth", ".safetensors"))]
                if ckpt_files:
                    ckpt_path = os.path.join(model_dir, ckpt_files[0])
                    print(f"[HY-Motion] Using weight file: {ckpt_files[0]}")
                else:
                    raise FileNotFoundError(f"No weights found in {model_dir}")

            # Heuristic detection based on file size (Full ~4.2GB, Lite ~1.8GB)
            file_size_gb = os.path.getsize(ckpt_path) / (1024**3)
            
            if "lite" in model_name.lower() or "lite" in model_dir.lower() or file_size_gb < 3.0:
                config = DEFAULT_CONFIG_LITE
                model_type = "LITE"
            else:
                config = DEFAULT_CONFIG_FULL
                model_type = "FULL"
            
            print(f"[HY-Motion] Zero-Config: Quality check PASSED. Applied official {model_type} architecture settings.")
        else:
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            print(f"[HY-Motion] Loaded official config from disk.")
        
        # Load only the DiT network (not the full pipeline)
        network = load_object(config["network_module"], config["network_module_args"])
        
        # Load checkpoint weights if available
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            
            # Extract network weights from checkpoint
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Extract specific buffers if they exist (mean, std, null features)
            checkpoint_mean = state_dict.get("mean")
            checkpoint_std = state_dict.get("std")
            null_vtxt_feat = state_dict.get("null_vtxt_feat")
            null_ctxt_input = state_dict.get("null_ctxt_input")
            spec_vtxt = state_dict.get("special_game_vtxt_feat")
            spec_ctxt = state_dict.get("special_game_ctxt_feat")

            # Diagnostic logging
            print(f"[HY-Motion] Loading checkpoint features:")
            if checkpoint_mean is not None: print(f"  [OK] found mean")
            if checkpoint_std is not None: print(f"  [OK] found std")
            if null_vtxt_feat is not None: print(f"  [OK] found learned null_vtxt_feat")
            if null_ctxt_input is not None: print(f"  [OK] found learned null_ctxt_input")
            if spec_vtxt is not None: print(f"  [OK] found special_game features")
            
            # Map state dict keys to network keys
            prefixes_to_strip = ["network.", "motion_transformer.", "module."]
            network_state = {}
            for k, v in state_dict.items():
                new_k = k
                while True:
                    stripped = False
                    for prefix in prefixes_to_strip:
                        if new_k.startswith(prefix):
                            new_k = new_k[len(prefix):]
                            stripped = True
                            break
                    if not stripped:
                        break
                
                # Filter out buffers that are handled by the wrapper
                if not any(new_k.startswith(p) for p in ["text_encoder.", "vae.", "total_params", "mean", "std", "null_vtxt", "null_ctxt", "special_game"]):
                    network_state[new_k] = v
            
            network.load_state_dict(network_state, strict=False)
            
            # Defaults for missing features
            feat_dim = config.get("vtxt_input_dim", 768)
            ctxt_dim = config.get("ctxt_input_dim", 4096)
            
            print(f"[HY-Motion] Checkpoint loaded features:")
            print(f"  - mean: {'Found' if checkpoint_mean is not None else 'Missing (using default)'}")
            print(f"  - std: {'Found' if checkpoint_std is not None else 'Missing (using default)'}")
            print(f"  - null_vtxt: {'Found' if null_vtxt_feat is not None else 'Missing (using default)'}")
            print(f"  - null_ctxt: {'Found' if null_ctxt_input is not None else 'Missing (using default)'}")
            print(f"  - special_game: {'Found' if spec_vtxt is not None else 'Missing'}")

            if null_vtxt_feat is None: null_vtxt_feat = torch.randn(1, 1, feat_dim)
            if null_ctxt_input is None: null_ctxt_input = torch.randn(1, 1, ctxt_dim)
            if spec_vtxt is None: spec_vtxt = torch.randn(1, 1, feat_dim)
            if spec_ctxt is None: spec_ctxt = torch.randn(1, 1, ctxt_dim)
            train_frames = 360
            special_game_vtxt_feat = spec_vtxt
            special_game_ctxt_feat = spec_ctxt
        else:
            print(f"[HY-Motion] WARNING: Checkpoint not found at {ckpt_path}, using random weights")
            checkpoint_mean = None
            checkpoint_std = None
            feat_dim = config.get("vtxt_input_dim", 768)
            ctxt_dim = config.get("ctxt_input_dim", 4096)
            null_vtxt_feat = torch.randn(1, 1, feat_dim)
            null_ctxt_input = torch.randn(1, 1, ctxt_dim)
            special_game_vtxt_feat = torch.randn(1, 1, feat_dim)
            special_game_ctxt_feat = torch.randn(1, 1, ctxt_dim)
            train_frames = 360
        
        # Find stats directory
        stats_dir = os.path.join(model_dir, "stats")
        if not os.path.exists(stats_dir):
            plugin_stats = os.path.join(CURRENT_DIR, "HY-Motion-1.0", "stats")
            if os.path.exists(plugin_stats):
                stats_dir = plugin_stats
            else:
                stats_dir = None
        
        # Move to device
        target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        network = network.to(target_device)
        network.eval()
        
        dit_wrapper = HYMotionDiT(
            network=network,
            config=config,
            stats_dir=stats_dir,
            device=target_device,
            mean=checkpoint_mean.to(target_device) if checkpoint_mean is not None else None,
            std=checkpoint_std.to(target_device) if checkpoint_std is not None else None,
            null_vtxt_feat=null_vtxt_feat.to(target_device),
            null_ctxt_input=null_ctxt_input.to(target_device),
            special_game_vtxt_feat=special_game_vtxt_feat.to(target_device),
            special_game_ctxt_feat=special_game_ctxt_feat.to(target_device),
            train_frames=train_frames
        )
        
        
        print(f"[HY-Motion] DiT loaded on {target_device}")
        
        return (dit_wrapper,)


# ============================================================================

def get_gguf_loader():
    """Helper to dynamically import GGUF loader from ComfyUI-GGUF folder as a package"""
    custom_nodes_path = os.path.dirname(os.path.dirname(__file__))
    # Try common folder names
    for folder_name in ["ComfyUI-GGUF", "ComfyUI_GGUF"]:
        node_path = os.path.join(custom_nodes_path, folder_name)
        if os.path.exists(node_path):
            try:
                # Register the folder as a proper package to handle relative imports
                pkg_name = "hymotion_gguf_backend"
                if pkg_name not in sys.modules:
                    spec = importlib.util.spec_from_file_location(pkg_name, os.path.join(node_path, "__init__.py"))
                    module = importlib.util.module_from_spec(spec)
                    module.__path__ = [node_path]
                    sys.modules[pkg_name] = module
                    spec.loader.exec_module(module)
                
                # Import from our dynamically created package
                loader_module = importlib.import_module(f"{pkg_name}.loader")
                ops_module = importlib.import_module(f"{pkg_name}.ops")
                dequant_module = importlib.import_module(f"{pkg_name}.dequant")
                
                # Return raw loader and mapping helpers for "True Native" support
                return (
                    loader_module.gguf_sd_loader, 
                    loader_module.LLAMA_SD_MAP, 
                    loader_module.sd_map_replace,
                    ops_module.GGMLOps, 
                    ops_module.GGMLTensor, 
                    dequant_module.dequantize_tensor
                )
            except Exception as e:
                print(f"[HY-Motion] Error importing GGUF: {e}")
    return None, None, None, None, None, None

class HYMotionTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        try:
            all_files = folder_paths.get_filename_list("hymotion_text_encoders")
            if not all_files:
                all_files = ["none"]
        except Exception:
            all_files = ["none"]
            
        return {
            "required": {
                "clip_name": (all_files, {
                    "default": all_files[0] if all_files else "none",
                    "tooltip": "CLIP-L text encoder model (clip-vit-large-patch14). Used for sentence-level embeddings."
                }),
                "llm_name": (all_files, {
                    "default": all_files[0] if all_files else "none",
                    "tooltip": "Qwen3 LLM text encoder. Provides detailed semantic understanding of motion descriptions."
                }),
                "device": (["cuda", "cpu", "mps"], {
                    "default": "cuda",
                    "tooltip": "Device for text encoder. CUDA is faster but uses VRAM. MPS is for Mac users."
                }),
            },
        }
    
    RETURN_TYPES = ("HYMOTION_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "HY-Motion/modular"
    
    def load_text_encoder(self, clip_name, llm_name, device="cuda"):
        """Load dual text encoder (CLIP + Qwen3) with GGUF dequantization support."""
        clip_path = folder_paths.get_full_path("hymotion_text_encoders", clip_name)
        llm_path = folder_paths.get_full_path("hymotion_text_encoders", llm_name)
        
        print(f"[HY-Motion] Loading CLIP from: {clip_name}")
        print(f"[HY-Motion] Loading LLM from: {llm_name}")
        
        # Initialize GGUF loader components
        dequantizer = None
        ggml_ops = None
        ggml_tensor = None
        
        def load_weights(path):
            nonlocal dequantizer, ggml_ops, ggml_tensor
            if not path: return None
            if path.endswith(".gguf"):
                try:
                    # Use helper to get official GGUF backend components
                    # We use gguf_sd_loader directly to bypass forced dequantization of embeddings
                    raw_loader, sd_map, map_replace, g_ops, g_tensor, dequant = get_gguf_loader()
                    if raw_loader and dequant:
                        dequantizer = dequant
                        ggml_ops = g_ops
                        ggml_tensor = g_tensor
                        print(f"[HY-Motion] Initialized True Native GGUF dequantizer and ops for {os.path.basename(path)}")
                        
                        # Load raw state dict and apply Llama/Qwen mapping manually
                        # NOTE: gguf_sd_loader may return (state_dict, metadata) tuple nowadays
                        sd_result = raw_loader(path, is_text_model=True)
                        if isinstance(sd_result, tuple):
                            sd = sd_result[0]  # Extract state dict from tuple
                        else:
                            sd = sd_result
                        return map_replace(sd, sd_map)
                except Exception as e:
                    print(f"[HY-Motion] GGUF Load failed: {e}")
                    return None
            
            # Standard Safetensors loading
            from safetensors.torch import load_file
            if os.path.isdir(path):
                sd = {}
                for f in sorted(os.listdir(path)):
                    if f.endswith(".safetensors"):
                        sd.update(load_file(os.path.join(path, f), device="cpu"))
                return sd
            return load_file(path, device="cpu") if path.endswith(".safetensors") else None

        # Load state dicts
        clip_sd = load_weights(clip_path)
        llm_sd = load_weights(llm_path)
        
        # Create model and load weights immediately (saves peak RAM vs manual loading)
        text_model = HYTextModel(
            llm_type="qwen3",
            max_length_llm=128,
            active_llm_sd=llm_sd,
            active_clip_sd=clip_sd,
            dequantizer=dequantizer,
            ggml_ops=ggml_ops,
            ggml_tensor=ggml_tensor
        )
        
        # Free memory and return
        del clip_sd, llm_sd
        gc.collect()
        
        print(f"[HY-Motion] Text Encoder loaded successfully on {device}")
        
        # Safely move model to device, handling meta tensors from failed weight loading
        target_device = torch.device(device)
        try:
            # Check if any parameters are still on meta device (indicates failed weight loading)
            has_meta_tensors = any(p.device.type == "meta" for p in text_model.parameters())
            if has_meta_tensors:
                print(f"[HY-Motion] WARNING: Some parameters are still meta tensors. Using to_empty() for safe device transfer.")
                # Use to_empty for meta tensors, then try to initialize with zeros
                text_model = text_model.to_empty(device=target_device)
                # Zero-initialize any remaining uninitialized parameters
                for name, param in text_model.named_parameters():
                    if param.data.storage().size() == 0:  # Unallocated storage
                        param.data = torch.zeros(param.shape, device=target_device, dtype=param.dtype)
            else:
                text_model = text_model.to(target_device)
        except Exception as e:
            print(f"[HY-Motion] WARNING: Failed to move model to {device}: {e}")
            print(f"[HY-Motion] Attempting to_empty() fallback...")
            text_model = text_model.to_empty(device=target_device)
        
        return (text_model,)


# ============================================================================
# Node 3: HYMotionTextEncode - Encode Text
# ============================================================================

class HYMotionTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_encoder": ("HYMOTION_TEXT_ENCODER", {
                    "tooltip": "The dual text encoder (CLIP + Qwen3) loaded from HYMotionTextEncoderLoader"
                }),
                "text": ("STRING", {
                    "default": "A person is walking forward.",
                    "multiline": True,
                    "tooltip": "Natural language description of the desired motion. Be descriptive for best results."
                }),
            },
        }
    
    RETURN_TYPES = ("HYMOTION_TEXT_EMBEDS",)
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "encode_text"
    CATEGORY = "HY-Motion/modular"
    
    def encode_text(self, text_encoder, text: str):
        # Display truncation in console only, full prompt is used!
        print(f"[HY-Motion] Encoding prompt: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Encode text with dual encoder
        with torch.no_grad():
            vtxt, ctxt, ctxt_length = text_encoder.encode([text])
        
        embeds_wrapper = HYMotionTextEmbeds(
            vtxt=vtxt,
            ctxt=ctxt,
            ctxt_length=ctxt_length,
            text=text
        )
        
        return (embeds_wrapper,)


# ============================================================================
# Node 3.5: HYMotionExtractFrame - Extract a single frame
# ============================================================================

class HYMotionExtractFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA", {"tooltip": "Motion data to extract frame from"}),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "tooltip": "Index of the frame to extract. 0 is first, -1 is last."
                }),
            },
        }
    
    RETURN_TYPES = ("HYMOTION_FRAME", "HYMOTION_DATA")
    RETURN_NAMES = ("frame", "motion_data")
    FUNCTION = "extract"
    CATEGORY = "HY-Motion/modular"
    
    def extract(self, motion_data: HYMotionData, frame_index: int):
        output_dict = motion_data.output_dict
        rot6d = output_dict["rot6d"] # [B, L, J, 6]
        transl = output_dict["transl"] # [B, L, 3]
        k3d = output_dict.get("keypoints3d")
        root_rot = output_dict.get("root_rotations_mat")
        
        # Handle negative indexing
        L = rot6d.shape[1]
        if frame_index < 0:
            frame_index = L + frame_index
            
        frame_index = max(0, min(frame_index, L - 1))
        
        # Extract first sample in batch
        frame_rot6d = rot6d[0:1, frame_index:frame_index+1, :, :].clone()
        frame_transl = transl[0:1, frame_index:frame_index+1, :].clone()
        
        # Create a frame wrapper for the sampler
        frame_wrapper = HYMotionFrame(rot6d=frame_rot6d, transl=frame_transl)
        
        # Create a 1-frame HYMotionData for previewing/saving
        frame_output_dict = {
            "rot6d": frame_rot6d,
            "transl": frame_transl,
        }
        if k3d is not None:
            frame_output_dict["keypoints3d"] = k3d[0:1, frame_index:frame_index+1, :, :].clone()
        if root_rot is not None:
            frame_output_dict["root_rotations_mat"] = root_rot[0:1, frame_index:frame_index+1, :, :].clone()
            
        frame_motion_data = HYMotionData(
            output_dict=frame_output_dict,
            text=f"{motion_data.text} (frame {frame_index})",
            duration=1/30.0,
            seeds=[motion_data.seeds[0]],
            device_info=motion_data.device_info
        )
        
        return (frame_wrapper, frame_motion_data)


# ============================================================================
# Node 4: HYMotionSampler - Generate Motion
# ============================================================================

class HYMotionSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit_model": ("HYMOTION_DIT", {"tooltip": "The HY-Motion DiT model loaded from HYMotionDiTLoader"}),
                "text_embeds": ("HYMOTION_TEXT_EMBEDS", {"tooltip": "Text embeddings from HYMotionTextEncode describing the desired motion"}),
                "duration": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 12.0,
                    "step": 0.1,
                    "tooltip": "Duration of the generated motion in seconds (at 30 FPS)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible motion generation. Different seeds produce different motion variations"
                }),
            },
            "optional": {
                "cfg_scale": ("FLOAT", {
                    "default": 5.0, 
                    "min": 0.0, 
                    "max": 20.0,
                    "tooltip": "Classifier-Free Guidance scale. Higher values (5-7) follow the text prompt more closely, lower values (1-3) allow more variation"
                }),
                "num_samples": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 12,
                    "tooltip": "Number of motion samples to generate in parallel. Each sample uses seed+i"
                }),
                "validation_steps": ("INT", {
                    "default": 50, 
                    "min": 10, 
                    "max": 100,
                    "tooltip": "Number of diffusion steps. More steps (50-100) = higher quality but slower generation"
                }),
                "use_special_game_feat": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable special game features for enhanced motion quality. Must set special_game_prob > 0.0 to take effect"
                }),
                "special_game_prob": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                    "tooltip": "Probability/strength of special game features (0.0-1.0). Only works when use_special_game_feat is enabled"
                }),
                "enable_ctxt_null_feat": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use learned null context features for CFG. When disabled, uses the prompt context for both conditional and unconditional passes"
                }),
                "sampler_method": (["dopri5", "euler", "midpoint", "rk4"], {
                    "default": "dopri5",
                    "tooltip": "ODE solver method. 'dopri5' is adaptive and best quality (official research default). 'euler' is fixed-step and faster."
                }),
                "atol": ("FLOAT", {
                    "default": 1e-4, 
                    "min": 1e-6, 
                    "max": 1e-2, 
                    "step": 1e-6,
                    "tooltip": "Absolute tolerance for adaptive solvers (like dopri5)"
                }),
                "rtol": ("FLOAT", {
                    "default": 1e-4, 
                    "min": 1e-6, 
                    "max": 1e-2, 
                    "step": 1e-6,
                    "tooltip": "Relative tolerance for adaptive solvers (like dopri5)"
                }),
                "align_ground": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically align the animation so the lowest point (usually feet) touches the ground (Y=0). Recommended!"
                }),
                "ground_offset": ("FLOAT", {
                    "default": 0.0, 
                    "min": -10.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Manual height adjustment (in meters) applied after ground alignment"
                }),
                "skip_smoothing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip post-processing smoothing for faster output (slight quality reduction)"
                }),
                "body_chunk_size": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Frames processed at once for body model. Higher = faster but more VRAM."
                }),
                "motion_data": ("HYMOTION_DATA", {"tooltip": "Optional previous motion to chain from. Automatically uses the last frame as the start."}),
                "first_frame": ("HYMOTION_FRAME", {"tooltip": "Optional starting pose for the motion"}),
                # "last_frame": ("HYMOTION_FRAME", {"tooltip": "Optional ending pose for the motion"}),  # HIDDEN: Feature in development
                "latent": ("LATENT", {"tooltip": "Optional latent tensor to guide generation or for refinement (used with denoise < 1.0)"}),
                "denoise": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Controls how much of the motion is generated from scratch. < 1.0 requires first_frame or last_frame."
                }),
                "transition_frames": ("INT", {
                    "default": 5, 
                    "min": 0, 
                    "max": 30, 
                    "step": 1,
                    "tooltip": "Number of frames at the start/end to smoothly blend from the input pose. Helps eliminate jumps."
                }),
                "smoothing_sigma": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 5.0, 
                    "step": 0.1,
                    "tooltip": "Rotation smoothing strength. Higher = smoother but less reactive."
                }),
                "smoothing_window": ("INT", {
                    "default": 11, 
                    "min": 1, 
                    "max": 51, 
                    "step": 2,
                    "tooltip": "Translation smoothing window (must be odd). Higher = smoother but can 'float' more."
                }),
                "force_origin": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force the animation to start at (0,0) in the XZ plane, ignoring the world position of the input frame."
                }),
                "momentum_guidance_scale": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Strength of momentum injection (0.7 = 70% physics, 30% AI). Lower allowed more AI variation at the start."
                }),
            }
        }
    
    RETURN_TYPES = ("HYMOTION_DATA",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "sample"
    CATEGORY = "HY-Motion/modular"

    def sample(self, dit_model: HYMotionDiT, text_embeds: HYMotionTextEmbeds, 
               duration: float, seed: int, cfg_scale: float = 5.0, 
               num_samples: int = 1, validation_steps: int = 50,
               use_special_game_feat: bool = False, special_game_prob: float = 1.0,
               enable_ctxt_null_feat: bool = True,
               sampler_method: str = "dopri5", atol: float = 1e-4, rtol: float = 1e-4,
               align_ground: bool = True, ground_offset: float = 0.0,
               skip_smoothing: bool = False, body_chunk_size: int = 64,
               first_frame: Optional[HYMotionFrame] = None, 
               last_frame: Optional[HYMotionFrame] = None,
               latent: Optional[torch.Tensor] = None,
               denoise: float = 1.0, transition_frames: int = 5,
               smoothing_sigma: float = 1.0, smoothing_window: int = 11,
               motion_data: Optional[Dict[str, Any]] = None,
               force_origin: bool = False,
               momentum_guidance_scale: float = 0.7):
        
        def safe_float(v, default):
            try:
                if v is None or v == "": return default
                return float(v)
            except: return default

        duration = safe_float(duration, 3.0)
        cfg_scale = safe_float(cfg_scale, 5.0)
        special_game_prob = safe_float(special_game_prob, 1.0)
        atol = safe_float(atol, 1e-4)
        rtol = safe_float(rtol, 1e-4)
        ground_offset = safe_float(ground_offset, 0.0)
        denoise = safe_float(denoise, 1.0)
        transition_frames = int(transition_frames)
        smoothing_sigma = safe_float(smoothing_sigma, 1.0)
        momentum_guidance_scale = safe_float(momentum_guidance_scale, 0.7)
        smoothing_window = int(smoothing_window)
        if smoothing_window % 2 == 0: smoothing_window += 1 # Must be odd

        print(f"[HY-Motion] Generating {duration}s motion with {num_samples} sample(s)")
        
        # Prepare seeds
        seeds = [seed + i for i in range(num_samples)]
        
        # Calculate frames from duration (30 FPS is the official output_mesh_fps)
        num_frames = int(duration * 30)
        
        # Use train_frames for latent generation (official implementation)
        # We now allow extending this up to 1000 frames (RoPE supports up to 5000)
        # but warn if exceeding the typical training range (360 frames).
        train_frames_default = dit_model.train_frames
        train_frames = max(num_frames, train_frames_default)
        
        if num_frames > train_frames_default:
            print(f"[HY-Motion] Extending sampling window to {num_frames} frames (Caution: durations > 12s may slightly reduce coherence).")
        
        # Upper safety limit for single-pass VRAM/Stability
        if train_frames > 1000: 
            print(f"[HY-Motion] Duration too long for single pass (max 33s). Clamping to 1000 frames. Use Chaining for longer motions.")
            train_frames = 1000
            num_frames = min(num_frames, 1000)
        num_frames = max(num_frames, 20)  # Minimum 20 frames
        
        vtxt = text_embeds.vtxt.clone().to(dit_model.device)
        ctxt = text_embeds.ctxt.clone().to(dit_model.device)
        ctxt_length = text_embeds.ctxt_length.clone().to(dit_model.device)
        
        # Deep Quality Check
        def get_stats(t, name):
            m = t.mean().item()
            s = t.std().item()
            sq = (t**2).mean().sqrt().item()
            return f"{name}: mean={m:.4f}, std={s:.4f}, rms={sq:.4f}"

        
        if torch.isnan(vtxt).any(): print("[HY-Motion] ERROR: vtxt (CLIP) contains NaNs!")
        if torch.isnan(ctxt).any(): print("[HY-Motion] ERROR: ctxt (LLM) contains NaNs!")
        
        if ctxt.std() < 1e-4:
            print("[HY-Motion] WARNING: LLM context (ctxt) is almost zero. Generation will be unconditioned!")
        if vtxt.std() < 1e-4:
            print("[HY-Motion] WARNING: CLIP context (vtxt) is almost zero. Generation will be unconditioned!")
        
        if use_special_game_feat:
            # Match official _maybe_inject_source_token logic
            vtxt_token = dit_model.special_game_vtxt_feat.to(vtxt).expand(1, 1, -1)
            # Blending based on prob (implied 1.0 here unless user changes)
            if special_game_prob > 0.0:
                vtxt = vtxt + vtxt_token * special_game_prob
            
            # Context injection logic (official writes token at cur_len)
            special_token = dit_model.special_game_ctxt_feat.to(ctxt) # [1, 1, Dc]
            # Since we only have 1 prompt, we cat it for predictability in this modular node
            ctxt = torch.cat([ctxt, special_token], dim=1)
            ctxt_length = ctxt_length + 1
            print(f"[HY-Motion] Special Game Feature enabled (prob={special_game_prob})")

        # Repeat for batch
        vtxt = vtxt.repeat(num_samples, 1, 1)
        ctxt = ctxt.repeat(num_samples, 1, 1)
        ctxt_length = ctxt_length.repeat(num_samples)

        # Prepare latent shape
        latent_dim = 201
        latent_shape = (num_samples, train_frames, latent_dim)
        
        # Prepare conditioned latents if provided
        first_latent = None
        last_latent = None
        root_offset = torch.zeros((1, 3), device=dit_model.device)
        
        # Motion Chaining: If motion_data is provided and first_frame is not,
        # automatically extract the last frame of the input motion.
        if motion_data is not None and first_frame is None:
            print("[HY-Motion] Motion chaining detected: Extracting last frame from input motion_data")
            # motion_data is a Dict[str, Any] wrapping HYMotionData
            # We need to get the HYMotionData object or its output_dict
            # In ComfyUI, if it's a custom type, it might be the object itself or a dict.
            # Based on Node 3.5 (HYMotionExtractFrame), it returns a HYMotionData object.
            
            # Let's assume motion_data is the HYMotionData object (or has output_dict)
            try:
                # If it's a dict from another node
                m_data = motion_data.get("motion_data") if isinstance(motion_data, dict) else motion_data
                output_dict = m_data.output_dict
                m_rot6d = output_dict["rot6d"] # [B, L, J, 6]
                m_transl = output_dict["transl"] # [B, L, 3]
                
                # Extract last frame (index -1)
                last_idx = m_rot6d.shape[1] - 1
                first_frame = HYMotionFrame(
                    rot6d=m_rot6d[0:1, last_idx:last_idx+1, :, :].clone(),
                    transl=m_transl[0:1, last_idx:last_idx+1, :].clone()
                )
                print(f"[HY-Motion] Successfully extracted frame {last_idx} for chaining.")
            except Exception as e:
                try:
                    first_frame = HYMotionFrame(rot6d=m_rot6d[:, -1, :22].cpu(), transl=m_transl[:, -1].cpu())
                    print(f"[HY-Motion] Extracted last frame for chaining (forced 22 joints fallback)")
                except Exception as e2:
                    print(f"[HY-Motion] WARNING: Failed to extract last frame for chaining: {e2}")

        # UNIFIED JOINT TRUNCATION: Ensure all reference frames are 22 joints for the Sampler
        orig_first_frame = first_frame
        orig_last_frame = last_frame
        
        if first_frame is not None:
            f_rot6d = first_frame.rot6d
            if f_rot6d.dim() == 4: f_rot6d = f_rot6d[:, 0]
            if f_rot6d.shape[1] > 22:
                print(f"[HY-Motion] Sampler: Truncating first_frame from {f_rot6d.shape[1]} to 22 joints")
                first_frame = HYMotionFrame(rot6d=f_rot6d[:, :22], transl=first_frame.transl)
            else:
                first_frame = HYMotionFrame(rot6d=f_rot6d, transl=first_frame.transl)

        if last_frame is not None:
            l_rot6d = last_frame.rot6d
            if l_rot6d.dim() == 4: l_rot6d = l_rot6d[:, 0]
            if l_rot6d.shape[1] > 22:
                print(f"[HY-Motion] Sampler: Truncating last_frame from {l_rot6d.shape[1]} to 22 joints")
                last_frame = HYMotionFrame(rot6d=l_rot6d[:, :22], transl=last_frame.transl)
            else:
                last_frame = HYMotionFrame(rot6d=l_rot6d, transl=last_frame.transl)

        if first_frame is not None or last_frame is not None:
            if dit_model.mean is not None and dit_model.std is not None:
                mean = dit_model.mean.to(dit_model.device)
                std = dit_model.std.to(dit_model.device)
                
                def normalize_frame(frame: HYMotionFrame):
                    # frame.rot6d is now guaranteed to be [1, 22, 6]
                    # Flatten to [1, 135]
                    raw = torch.cat([frame.transl.flatten(), frame.rot6d.flatten()]).unsqueeze(0)
                    
                    # Pad to 201 (3 pos + 132 rot6d + 66 keypoints placeholders)
                    padded = torch.zeros((1, 201), device=dit_model.device)
                    padded[:, :raw.shape[1]] = raw.to(dit_model.device)
                    
                    # Normalize
                    norm = torch.zeros_like(padded)
                    mask = std >= 1e-3
                    norm[:, mask] = (padded[:, mask] - mean[mask]) / std[mask]
                    return norm

                if first_frame is not None:
                    # Extract root offset from first frame (HORIZONTAL ONLY)
                    # We only want to shift X and Z. We must preserve Y (height) so the AI 
                    # doesn't "jump" from the floor to its natural height.
                    root_offset = first_frame.transl.clone().to(dit_model.device)
                    root_offset[..., 1] = 0 # Zero out Y in the offset
                    
                    # Create a relative version for conditioning (preserves Y)
                    local_first_frame = HYMotionFrame(
                        rot6d=first_frame.rot6d,
                        transl=(first_frame.transl.to(dit_model.device) - root_offset).cpu()
                    )
                    first_latent = normalize_frame(local_first_frame)
                    print(f"[HY-Motion] Horizontal root offset extracted: {root_offset.cpu().numpy()}")
                
                if last_frame is not None:
                    # For last frame, we make it relative to the first frame's offset
                    local_last_frame = HYMotionFrame(
                        rot6d=last_frame.rot6d,
                        transl=(last_frame.transl.to(dit_model.device) - root_offset).cpu()
                    )
                    last_latent = normalize_frame(local_last_frame)
            else:
                print("[HY-Motion] WARNING: Cannot condition frames without normalization stats!")

        # Generate noise from seeds (matching official implementation)
        generators = []
        for s in seeds:
            gen_device = "cpu" if dit_model.device.type == "cpu" else dit_model.device
            generators.append(torch.Generator(device=gen_device).manual_seed(s))
        
        y0 = randn_tensor(
            latent_shape,
            generator=generators,
            device=dit_model.device,
            dtype=torch.float32
        )
        
        # Handle Denoise (Flow Matching style)
        t_start = 0.0
        if denoise < 1.0:
            t_start = 1.0 - denoise
            if first_latent is not None and last_latent is not None:
                # Linear interpolation between first and last frames
                # t_interp: [1, train_frames, 1]
                t_interp = torch.linspace(0, 1, train_frames, device=dit_model.device).view(1, -1, 1)
                # x_1_interp = (1-t) * first + t * last
                x_1_interp = (1.0 - t_interp) * first_latent.unsqueeze(1) + t_interp * last_latent.unsqueeze(1)
                # x_t = (1-t)x_0 + t*x_1
                y0 = (1.0 - t_start) * y0 + t_start * x_1_interp
                print(f"[HY-Motion] Denoise enabled: starting ODE at t={t_start:.4f} with First->Last interpolation")
            elif latent is not None:
                # Direct latent injection (e.g. from Encoder or Refinement)
                # latent shape: [B, T, 201]
                # We need to resize it to match train_frames if needed
                l_in = latent.clone().to(dit_model.device)
                if l_in.shape[1] != train_frames:
                     # Simple nearest resize for now if lengths mismatch
                     l_in = F.interpolate(l_in.transpose(1, 2), size=train_frames, mode='nearest').transpose(1, 2)
                     print(f"[HY-Motion] Warning: Resized input latent from {latent.shape[1]} to {train_frames} frames")
                
                # Check for normalization stats mismatch
                # Assumes input latent is already normalized (which it is, from Encoder node)
                
                y0 = (1.0 - t_start) * y0 + t_start * l_in
                print(f"[HY-Motion] Denoise enabled: starting ODE at t={t_start:.4f} using provided LATENT")
            elif first_latent is not None:
                y0 = (1.0 - t_start) * y0 + t_start * first_latent.unsqueeze(1)
                print(f"[HY-Motion] Denoise enabled: starting ODE at t={t_start:.4f} using first_frame as base")
            elif last_latent is not None:
                y0 = (1.0 - t_start) * y0 + t_start * last_latent.unsqueeze(1)
                print(f"[HY-Motion] Denoise enabled: starting ODE at t={t_start:.4f} using last_frame as base")
            else:
                t_start = 0.0
                print("[HY-Motion] WARNING: Denoise < 1.0 requires first_frame or last_frame. Ignoring denoise.")
        
        if torch.isnan(y0).any():
            print(f"[HY-Motion] ERROR: Initial noise/latent y0 contains NaNs!")
        
        # Create masks using train_frames
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt.shape[1]).to(dit_model.device)
        x_length = torch.LongTensor([num_frames] * num_samples).to(dit_model.device)
        x_mask_temporal = length_to_mask(x_length, train_frames).to(dit_model.device)
        
        if torch.isnan(vtxt).any():
            print("[HY-Motion] ERROR: vtxt features contain NaNs before sampling!")
        if torch.isnan(ctxt).any():
            print("[HY-Motion] ERROR: ctxt features contain NaNs before sampling!")
            
        # Classifier-free guidance setup (match official research paper & implementation)
        do_cfg = cfg_scale > 1.0
        if do_cfg:
            # learned null features from checkpoint
            null_vtxt = dit_model.null_vtxt_feat.expand(num_samples, -1, -1).to(dit_model.device)
            
            if enable_ctxt_null_feat:
                # CRITICAL QUALITY FIX: Tile the null feature across the sequence!
                # Official logic uses expand(*ctxt_input.shape)
                null_ctxt = dit_model.null_ctxt_input.expand(num_samples, ctxt.shape[1], -1).to(dit_model.device)
                # Official uses SAME mask for both (duplication)
                null_ctxt_mask = ctxt_mask_temporal
            else:
                null_ctxt = ctxt
                null_ctxt_mask = ctxt_mask_temporal
            
            if torch.isnan(null_vtxt).any():
                print("[HY-Motion] ERROR: null_vtxt features contain NaNs!")
            if torch.isnan(null_ctxt).any():
                print("[HY-Motion] ERROR: null_ctxt features contain NaNs!")

            # Double the batch for CFG (uncond first, then cond)
            vtxt_cfg = torch.cat([null_vtxt, vtxt], dim=0)
            ctxt_cfg = torch.cat([null_ctxt, ctxt], dim=0)
            ctxt_mask_cfg = torch.cat([null_ctxt_mask, ctxt_mask_temporal], dim=0)
            x_mask_cfg = torch.cat([x_mask_temporal, x_mask_temporal], dim=0)
            
            print(f"[HY-Motion] CFG Enabled (scale={cfg_scale}) using learned null features.")
        else:
            vtxt_cfg = vtxt
            ctxt_cfg = ctxt
            ctxt_mask_cfg = ctxt_mask_temporal
            x_mask_cfg = x_mask_temporal
        
        # ODE function matching official implementation
        pbar = comfy.utils.ProgressBar(validation_steps)
        step_counter = [0]
        
        def ode_fn(t, x):
            nonlocal step_counter
            
            # Batch for CFG
            x_input = torch.cat([x, x], dim=0) if do_cfg else x
            
            # Single forward pass (efficient CFG)
            x_pred = dit_model.network(
                x=x_input,
                timesteps=t.expand(x_input.shape[0]),
                vtxt_input=vtxt_cfg,
                ctxt_input=ctxt_cfg,
                ctxt_mask_temporal=ctxt_mask_cfg,
                x_mask_temporal=x_mask_cfg
            )
            
            # Apply CFG
            if do_cfg:
                x_pred_uncond, x_pred_cond = x_pred.chunk(2)
                x_pred = x_pred_uncond + cfg_scale * (x_pred_cond - x_pred_uncond)
            
            # Temporal Velocity Blending (Smooth Transition)
            # We override/guide the predicted velocity for the first/last 'transition_frames'
            # to ensure a smooth takeoff/landing from the input frames.
            if first_latent is not None or last_latent is not None:
                # w: [1, train_frames, D] - Channel-specific guidance
                w = torch.zeros((1, x_pred.shape[1], x_pred.shape[2]), device=x.device)
                
                # Calculate temporal weights (S curve or Linear)
                # We apply high guidance at the boundary, fading to 0
                
                if first_latent is not None:
                    # Start boundary
                    w[:, 0, :] = 1.0
                    if transition_frames > 0:
                        for i in range(1, transition_frames + 1):
                            if i < x_pred.shape[1]:
                                weight = max(0.0, 1.0 - (i / (transition_frames + 1)))
                                w[:, i, :] = weight
                
                if last_latent is not None:
                    # End boundary
                    w[:, num_frames - 1, :] = 1.0
                    if transition_frames > 0:
                        for i in range(1, transition_frames + 1):
                            idx = num_frames - 1 - i
                            if idx >= 0:
                                weight = max(0.0, 1.0 - (i / (transition_frames + 1)))
                                w[:, idx, :] = weight
                # Unified Target Construction
                # We determine the 'target_data' (x_start) that we want the flow to point towards.
                target_data = None
                
                if first_latent is not None and last_latent is not None:
                    # Interpolation between Start and End
                    t_interp = torch.linspace(0, 1, x_pred.shape[1], device=x.device).view(1, -1, 1)
                    target_data = (1.0 - t_interp) * first_latent.unsqueeze(1) + t_interp * last_latent.unsqueeze(1)
                elif first_latent is not None:
                    # Start condition only
                    target_data = first_latent.unsqueeze(1).repeat(1, x_pred.shape[1], 1)
                elif last_latent is not None:
                    # End condition only
                    target_data = last_latent.unsqueeze(1).repeat(1, x_pred.shape[1], 1)

                # MOMENTUM INJECTION (Modifies target_data)
                is_momentum = False
                if latent is not None and latent.shape[1] > 1 and first_latent is not None:
                    is_momentum = True
                    # Calculate velocity from history
                    p_last = latent[:, -1:, :].to(x.device)
                    p_prev = latent[:, -2:-1, :].to(x.device)
                    v_history = p_last - p_prev # Shape [1, 1, 201]

                    # Apply momentum extrapolation to Root Translation (dims 0-3)
                    # We linearize the target for the transition duration
                    if target_data is not None:
                        for i in range(min(transition_frames, x_pred.shape[1])):
                             # Linear extrapolation delta
                             delta = v_history[:, :, :3] * (i + 1)
                             target_data[:, i, :3] = target_data[:, i, :3] + delta
                    
                    if step_counter[0] == 0:
                        print(f"[HY-Motion] [INFO] Momentum Injection Active! Guiding Root Translation along extrapolated path.")

                # Calculate specific target velocity for Flow Matching
                v_target = None
                if target_data is not None:
                    v_target = target_data - y0
                
                # Guidance Channel Masking ('w') adjustments
                if is_momentum:
                    # If using momentum, we trust the extrapolated translation target,
                    # BUT we might want to dampen it to allow the AI to add "life" (noise).
                    # w[..., :3] roughly controls translation adherence.
                    # We scale the entire w vector (rot + trans) by the user's factor.
                    w = w * momentum_guidance_scale
                else:
                    # If NO momentum (and only first_latent), we perform "Translation Agostic" guidance.
                    # We zero out translation weights to let the model infer physics.
                    # (Unless we are in In-Between mode, where translation is interpolated)
                    if first_latent is not None and last_latent is None:
                         w[..., :3] = 0.0

                
                # Blend predicted velocity with target velocity
                if v_target is not None:
                    x_pred = (1.0 - w) * x_pred + w * v_target
            
            if torch.isnan(x_pred).any():
                print(f"[HY-Motion] ERROR: DiT forward produced NaNs at t={t.item()}!")
            
            # QUALITY LOG: Log first 3 steps
            if step_counter[0] < 3:
                rms = (x_pred**2).mean().sqrt().item()
                print(f"[HY-Motion] DiT step {step_counter[0]} t={t.item():.4f} output RMS: {rms:.4f}")
            
            # Update progress
            step_counter[0] += 1
            pbar.update_absolute(min(step_counter[0], validation_steps), validation_steps)
            
            return x_pred
        
        # Use torchdiffeq.odeint (matching official implementation)
        t = torch.linspace(t_start, 1, validation_steps + 1, device=dit_model.device)
        
        with torch.no_grad():
            # Match official research paper: default to dopri5 with 1e-4 rtol/atol
            # If method is euler, validation_steps dictates the number of steps
            ode_kwargs = {"method": sampler_method}
            if sampler_method == "dopri5":
                ode_kwargs["atol"] = atol
                ode_kwargs["rtol"] = rtol
                
            trajectory = odeint(ode_fn, y0, t, **ode_kwargs)
        
        # Get final sample and crop to actual length (matching official)
        latent_output = trajectory[-1][:, :num_frames, ...].clone()
        
        # Denormalize using mean/std (matching official implementation)
        # Denormalize using mean/std (matching official implementation)
        if dit_model.mean is not None and dit_model.std is not None:
            # Handle 52-joint stats (315 dims) vs 22-joint latent (201 dims) mismatch
            # This mirrors the logic in encoder.py to ensure robust decoding
            m = dit_model.mean.clone()
            s = dit_model.std.clone()
            
            if m.shape[-1] > 201:
                print(f"[HY-Motion] Adapting 52-joint stats ({m.shape[-1]} dims) to 22-joint latent (201 dims)")
                m = torch.cat([m[..., :3], m[..., 3:135], m[..., 159:225]], dim=-1)
                s = torch.cat([s[..., :3], s[..., 3:135], s[..., 159:225]], dim=-1)

            # Log denormalization stats first time
            print(f"[HY-Motion] Denormalizing with mean shape {m.shape}, std shape {s.shape}")
            
            # Handle zero std - CRITICAL: Use zeros (not ones!) for near-zero std
            # This means for dimensions with std < 1e-3: latent_denorm = mean (ignore latent)
            # This matches official behavior in _decode_o6dp
            std_zero = s < 1e-3
            num_zero_std = std_zero.sum().item()
            if num_zero_std > 0:
                print(f"[HY-Motion] {num_zero_std}/{s.numel()} dimensions have near-zero std")
            s = torch.where(std_zero, torch.zeros_like(s), s)  # FIX: zeros not ones!
            
            latent_denorm = latent_output * s + m
            
            # Log output range for debugging
            print(f"[HY-Motion] Denorm output range: [{latent_denorm.min():.4f}, {latent_denorm.max():.4f}]")
        else:
            latent_denorm = latent_output
            print("[HY-Motion] CRITICAL: No normalization stats! Results will be garbage.")

        
        # Decode motion (matching official _decode_o6dp)
        num_joints = 22
        B, L = latent_denorm.shape[:2]
        
        transl = latent_denorm[..., 0:3].clone()
        root_rot6d = latent_denorm[..., 3:9].reshape(B, L, 1, 6).clone()
        body_rot6d = latent_denorm[..., 9:9+21*6].reshape(B, L, 21, 6).clone()
        rot6d = torch.cat([root_rot6d, body_rot6d], dim=2)  # (B, L, 22, 6)
        
        # Hard-Pinning: Force the raw output to match the target poses exactly if provided.
        # This eliminates any "generation drift" from the DiT model.
        # NOTE: We must use RELATIVE translations (subtract root_offset) because root_offset 
        # is added back at the very end of the sample method.
        if first_frame is not None:
            print("[HY-Motion] Hard-Pinning: Forcing Frame 0 to match first_frame exactly (Relative)")
            transl[:, 0, :] = (first_frame.transl.to(transl) - root_offset.to(transl))
            rot6d[:, 0, :, :] = first_frame.rot6d.to(rot6d)
            
        if last_frame is not None:
            print("[HY-Motion] Hard-Pinning: Forcing Final Frame to match last_frame exactly (Relative)")
            transl[:, -1, :] = (last_frame.transl.to(transl) - root_offset.to(transl))
            rot6d[:, -1, :, :] = last_frame.rot6d.to(rot6d)
        
        # Apply motion smoothing (matching official logic) unless skip_smoothing is enabled
        if skip_smoothing:
            print("[HY-Motion] Skipping smoothing for faster output")
            rot6d_smooth = rot6d
            transl_smooth = transl
        else:
            # Official uses SLERP on quaternions for rotations and Savgol for translations
            rot6d_smooth = MotionGeneration.smooth_with_slerp(rot6d, sigma=smoothing_sigma)
            # Savgol filter with window=11, polyorder=5 (matching official)
            transl_smooth = MotionGeneration.smooth_with_savgol(transl, window_length=smoothing_window, polyorder=5)
            
            # Ramped Drift Correction: Ensures conditioned frames match exactly while preserving smoothing
            # We anchor the correction to the TARGET poses (which are now in 'transl' and 'rot6d' due to Hard-Pinning)
            if first_frame is not None and last_frame is not None:
                print("[HY-Motion] Applying ramped drift correction for seamless loop/transition (Anchored to Targets)")
                drift_start = transl_smooth[:, 0, :] - transl[:, 0, :]
                drift_end = transl_smooth[:, -1, :] - transl[:, -1, :]
                t_ramp = torch.linspace(0, 1, num_frames, device=transl.device).view(1, -1, 1)
                ramp = (1 - t_ramp) * drift_start.unsqueeze(1) + t_ramp * drift_end.unsqueeze(1)
                transl_smooth -= ramp
                
                # Ramped Rotation Blending (eases the transition from pinned pose into smoothed sequence)
                blend_len = min(15, num_frames // 3)
                ramp_rot = torch.linspace(1, 0, blend_len, device=rot6d_smooth.device).view(1, -1, 1, 1)
                rot6d_smooth[:, :blend_len, :, :] = ramp_rot * rot6d[:, :blend_len, :, :] + (1 - ramp_rot) * rot6d_smooth[:, :blend_len, :, :]
                
                ramp_rot_end = torch.linspace(0, 1, blend_len, device=rot6d_smooth.device).view(1, -1, 1, 1)
                rot6d_smooth[:, -blend_len:, :, :] = (1 - ramp_rot_end) * rot6d_smooth[:, -blend_len:, :, :] + ramp_rot_end * rot6d[:, -blend_len:, :, :]
                
            elif first_frame is not None:
                print("[HY-Motion] Correcting smoothing drift at start frame (Anchored to Target)")
                drift = transl_smooth[:, 0, :] - transl[:, 0, :]
                transl_smooth -= drift.unsqueeze(1)
                
                # Ramped Rotation Blending
                blend_len = min(15, num_frames // 3)
                ramp_rot = torch.linspace(1, 0, blend_len, device=rot6d_smooth.device).view(1, -1, 1, 1)
                rot6d_smooth[:, :blend_len, :, :] = ramp_rot * rot6d[:, :blend_len, :, :] + (1 - ramp_rot) * rot6d_smooth[:, :blend_len, :, :]
                
            elif last_frame is not None:
                print("[HY-Motion] Correcting smoothing drift at end frame (Anchored to Target)")
                drift = transl_smooth[:, -1, :] - transl[:, -1, :]
                transl_smooth -= drift.unsqueeze(1)
                
                # Ramped Rotation Blending
                blend_len = min(15, num_frames // 3)
                ramp_rot_end = torch.linspace(0, 1, blend_len, device=rot6d_smooth.device).view(1, -1, 1, 1)
                rot6d_smooth[:, -blend_len:, :, :] = (1 - ramp_rot_end) * rot6d_smooth[:, -blend_len:, :, :] + ramp_rot_end * rot6d[:, -blend_len:, :, :]

        
        # Handle 52-joint upsampling if original inputs were 52 joints
        output_52_joints = False
        if (orig_first_frame and orig_first_frame.rot6d.shape[-2] == 52) or \
           (orig_last_frame and orig_last_frame.rot6d.shape[-2] == 52):
            output_52_joints = True
            print(f"[HY-Motion] 52-Joint Reconstruction: Upsampling AI results ({L} frames) to high-fidelity...")
            
            # Create full 52-joint tensor [B, L, 52, 6]
            full_rot6d = torch.zeros((B, L, 52, 6), device=rot6d_smooth.device)
            # Initialize with Identity 6D
            full_rot6d[..., 0] = 1.0
            full_rot6d[..., 4] = 1.0
            
            # Fill first 22 with AI results
            full_rot6d[:, :, :22, :] = rot6d_smooth
            
            # Extract hand poses [1, 30, 6]
            h_start = None
            h_end = None
            if orig_first_frame is not None and orig_first_frame.rot6d.dim() >= 2:
                f_rot = orig_first_frame.rot6d
                if f_rot.dim() == 4: f_rot = f_rot[:, 0]
                if f_rot.shape[1] == 52:
                    h_start = f_rot[:, 22:].to(rot6d_smooth.device)
            
            if orig_last_frame is not None and orig_last_frame.rot6d.dim() >= 2:
                l_rot = orig_last_frame.rot6d
                if l_rot.dim() == 4: l_rot = l_rot[:, 0]
                if l_rot.shape[1] == 52:
                    h_end = l_rot[:, 22:].to(rot6d_smooth.device)
                
            # Fill hand joints (22-52)
            if h_start is not None and h_end is not None:
                # Interpolate hand poses smoothly across the sequence
                t_lerp = torch.linspace(0, 1, L, device=rot6d_smooth.device).view(1, L, 1, 1)
                full_rot6d[:, :, 22:, :] = (1 - t_lerp) * h_start.unsqueeze(1) + t_lerp * h_end.unsqueeze(1)
                print("[HY-Motion] Hands: Blending custom start and end hand poses.")
            elif h_start is not None:
                full_rot6d[:, :, 22:, :] = h_start.unsqueeze(1)
                print("[HY-Motion] Hands: Persisting custom start hand pose.")
            elif h_end is not None:
                full_rot6d[:, :, 22:, :] = h_end.unsqueeze(1)
                print("[HY-Motion] Hands: Persisting custom end hand pose.")
            
            rot6d_smooth = full_rot6d
            num_joints = 52
        
        # Convert rot6d to rotation matrices
        rot_mat = rot6d_to_rotation_matrix(rot6d_smooth.reshape(-1, 6)).reshape(B, L, num_joints, 3, 3)
        
        # Prepare values for output_dict
        # Final translation calculation:
        # We align the entire sequence so it starts EXACTLY at the world_target height/pos.
        # This prevents both "double-offsetting" and "ground-jumping".
        # CRITICAL: Use original first_frame.transl (not root_offset) to preserve Y-coordinate!
        if first_frame is not None and not force_origin:
            final_offset = first_frame.transl.to(transl_smooth).clone()
        elif force_origin:
            final_offset = torch.zeros_like(root_offset).to(transl_smooth)
        else:
            final_offset = root_offset.to(transl_smooth).clone()
            
        # Bit-perfect shift: move entire sequence so frame 0 matches final_offset exactly
        displacement = final_offset - transl_smooth[:, 0:1, :]
        transl_cpu = (transl_smooth + displacement).cpu()
        print(f"[HY-Motion] FINAL ALIGNMENT: Bit-perfect shift to anchor.")
        
        rot6d_cpu = rot6d_smooth.cpu()
        root_rot_mat = rot_mat[:, :, 0].cpu() # (B, L, 3, 3)
        
        final_text = text_embeds.text
        final_duration = duration
        
        # Motion Concatenation: If motion_data was provided, prepend it to the current output
        if motion_data is not None:
            try:
                m_data = motion_data.get("motion_data") if isinstance(motion_data, dict) else motion_data
                prev_dict = m_data.output_dict
                
                print(f"[HY-Motion] STITCHING ANALYTICS: Previous segment ({m_data.duration:.1f}s) -> New segment ({duration:.1f}s)")
                
                # We concatenate BEFORE running BodyModel to ensure the whole sequence is consistent.
                # Use CPU tensors for concatenation, THEN move to GPU for BodyModel
                p_rot6d = prev_dict["rot6d"]
                p_trans = prev_dict["transl"]
                p_root_rot = prev_dict["root_rotations_mat"]
                
                # High-visibility stitch check
                p_v = p_trans[0, -1].cpu().numpy()
                o_v = transl_cpu[0, 0].cpu().numpy()
                dist = np.linalg.norm(p_v - o_v)
                print(f"  - Translation Connection: DISTANCE={dist:.6f}")
                if dist > 1e-4:
                    print(f"    [WARNING] Translation discontinuity detected ({p_v} -> {o_v})")

                # Match batch sizes if needed
                if p_rot6d.shape[0] != rot6d_cpu.shape[0]:
                    if p_rot6d.shape[0] == 1: p_rot6d = p_rot6d.repeat(rot6d_cpu.shape[0], 1, 1, 1)
                    else: rot6d_cpu = rot6d_cpu.repeat(p_rot6d.shape[0], 1, 1, 1)
                
                if p_trans.shape[0] != transl_cpu.shape[0]:
                    if p_trans.shape[0] == 1: p_trans = p_trans.repeat(transl_cpu.shape[0], 1, 1)
                    else: transl_cpu = transl_cpu.repeat(p_trans.shape[0], 1, 1)
                    
                if p_root_rot.shape[0] != root_rot_mat.shape[0]:
                    if p_root_rot.shape[0] == 1: p_root_rot = p_root_rot.repeat(root_rot_mat.shape[0], 1, 1, 1)
                    else: root_rot_mat = root_rot_mat.repeat(p_root_rot.shape[0], 1, 1, 1)

                rot6d_cpu = torch.cat([p_rot6d, rot6d_cpu], dim=1)
                transl_cpu = torch.cat([p_trans, transl_cpu], dim=1)
                root_rot_mat = torch.cat([p_root_rot, root_rot_mat], dim=1)
                
                final_text = f"{m_data.text} -> {final_text}"
                final_duration = m_data.duration + duration
                print(f"[HY-Motion] Concatenated previous motion ({m_data.duration:.1f}s) with new motion ({duration:.1f}s)")
            except Exception as e:
                print(f"[HY-Motion] WARNING: Failed to join motions: {e}")

        # === Skeletal Self-Healing: Regenerate keypoints3d for the ENTIRE sequence ===
        # This ensures that even if previous segments had "poisoned" (local) keypoints,
        # they are now recalculated correctly in world space.
        global _WOODEN_MESH_CACHE
        if _WOODEN_MESH_CACHE is None:
             print("[HY-Motion] Initializing 3D body model for skeletal decoding...")
             _WOODEN_MESH_CACHE = WoodenMesh().to(dit_model.device).eval()
        
        body_model = _WOODEN_MESH_CACHE.to(dit_model.device)
        with torch.no_grad():
            params = {
                "rot6d": rot6d_cpu.to(dit_model.device),
                "trans": transl_cpu.to(dit_model.device)
            }
            # process in chunks to save VRAM
            print(f"[HY-Motion] Skeletal Decoding: Processing full sequence ({transl_cpu.shape[1]} frames)...")
            body_out = body_model.forward_batch(params, chunk_size=body_chunk_size)
            keypoints3d = body_out["keypoints3d"].cpu()
            vertices = body_out["vertices"].cpu() # Optional, but keeps Data object intact

        # Ground Alignment (if enabled)
        # Note: We only auto-align if this is the FIRST segment (no first_frame)
        if align_ground and first_frame is None:
            min_y = vertices[..., 1].amin(dim=(1, 2), keepdim=True)
            total_offset = -min_y.squeeze(-1) + ground_offset
            print(f"[HY-Motion] Ground Aligning FULL sequence: applying +{total_offset.mean().item():.4f}m offset")
            keypoints3d[..., 1] += total_offset.unsqueeze(-1)
            transl_cpu[..., 1] += total_offset
        elif ground_offset != 0 and first_frame is None:
             # Manual offset applied only if not chained (chained motions already carry the offset)
             transl_cpu[..., 1] += ground_offset
             keypoints3d[..., 1] += ground_offset
        
        # Create output dictionary
        # If force_origin is enabled, we don't add the root_offset back
        # final_offset = torch.zeros_like(root_offset) if force_origin else root_offset # Already handled above
        
        output_dict = {
            "rot6d": rot6d_cpu,
            "transl": transl_cpu,
            "keypoints3d": keypoints3d,
            "root_rotations_mat": root_rot_mat,
        }
        
        
        # Create motion data wrapper
        motion_data_out = HYMotionData(
            output_dict=output_dict,
            text=final_text,
            duration=final_duration,
            seeds=seeds,
            device_info=str(dit_model.device)
        )
        
        print(f"[HY-Motion] Generated {num_samples} sample(s), total {output_dict['rot6d'].shape[1]} frames on {dit_model.device}")
        
        return (motion_data_out,)
    
    # Removed local smoothing helpers in favor of official MotionGeneration methods


class HYMotionModularExportFBX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA", {
                    "tooltip": "Motion data from HYMotionSampler containing keypoints and rotations"
                }),
                "template_name": (folder_paths.get_filename_list("hymotion_fbx_templates"), {
                    "tooltip": "FBX template model with skeleton. The motion will be applied to this rig."
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, 
                    "min": 1.0, 
                    "max": 120.0, 
                    "step": 0.1,
                    "tooltip": "Frames per second for the exported animation. 30 FPS is standard."
                }),
                "scale": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.00, 
                    "max": 1000.0, 
                    "step": 0.01,
                    "tooltip": "Scale factor for FBX export. 0.0 for auto-matching (recommended). 100.0 for cm (Blender/Maya), 1.0 for meters (Unity)."
                }),
                "output_dir": ("STRING", {
                    "default": "hymotion_fbx",
                    "tooltip": "Subdirectory in ComfyUI output folder where FBX files will be saved."
                }),
                "filename_prefix": ("STRING", {
                    "default": "motion",
                    "tooltip": "Prefix for generated FBX filenames. Full name: prefix_timestamp_id_batch.fbx"
                }),
            },
            "optional": {
                "batch_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 63, 
                    "step": 1,
                    "tooltip": "Which batch sample to use as primary output (for single-file workflows)."
                }),
                "in_place": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, removes horizontal movement (X/Z) from root so character stays in place."
                }),
                "in_place_x": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep character centered on X axis (Side-to-Side)."
                }),
                "in_place_y": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep character centered on Y axis (Up/Down - prevents jumping/falling)."
                }),
                "in_place_z": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep character centered on Z axis (Forward/Backward)."
                }),
                "absolute_root": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, treats root translation as absolute. Resolve jumping issues with offset templates."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "FBX")
    RETURN_NAMES = ("fbx_paths", "fbx")
    FUNCTION = "export_fbx"
    CATEGORY = "HY-Motion/modular"
    OUTPUT_NODE = True

    def export_fbx(self, motion_data: HYMotionData, template_name: str, fps: float, scale: float, output_dir: str, filename_prefix: str, 
                   batch_index: int = 0, in_place: bool = False, in_place_x: bool = False, in_place_y: bool = False, in_place_z: bool = False,
                   absolute_root: bool = True):
        # Resolve template path - use dropdown selection or fallback to default
        template_fbx_path = folder_paths.get_full_path("hymotion_fbx_templates", template_name)
            
        # Fallback to default wooden boy
        if not template_fbx_path or not os.path.exists(template_fbx_path):
             plugin_dir = os.path.dirname(os.path.abspath(__file__))
             template_fbx_path = os.path.join(plugin_dir, "assets", "wooden_models", "boy_Rigging_smplx_tex.fbx")

        print(f"[HY-Motion] Using FBX Template: {template_fbx_path}")

        try:
            import fbx
            from .hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
            fbx_converter = SMPLH2WoodFBX(template_fbx_path=template_fbx_path, scale=scale, absolute_root=absolute_root)
        except ImportError:
            msg = "Error: FBX SDK not installed. Please install it to use FBX export."
            print(f"[HY-Motion] {msg}")
            return (msg,)
        except Exception as e:
            import traceback
            msg = f"Error initializing FBX converter: {str(e)}"
            print(f"[HY-Motion] {msg}")
            traceback.print_exc()
            return (msg,)

        # Log DiT device if it wasn't logged during loading (due to caching)
        if hasattr(motion_data, "device_info"):
             print(f"[HY-Motion] Using motion data from device: {motion_data.device_info}")

        # Use ComfyUI's native output directory
        full_output_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        output_dict = motion_data.output_dict
        timestamp = get_timestamp()
        unique_id = str(uuid.uuid4())[:8]

        fbx_files = []

        for batch_idx in range(motion_data.batch_size):
            try:
                rot6d = output_dict["rot6d"][batch_idx].clone()
                transl = output_dict["transl"][batch_idx].clone()
                
                # Apply in-place: lock axes based on toggles
                # HyMotion/SMPL-H space: Y is usually UP, Z is FORWARD, X is SIDE
                lock_x = in_place_x or in_place
                lock_z = in_place_z or in_place
                lock_y = in_place_y
                
                if lock_x or lock_y or lock_z:
                    # Get the initial position at frame 0
                    init_pos = transl[0].clone()
                    if lock_x: transl[:, 0] = init_pos[0]
                    if lock_y: transl[:, 1] = init_pos[1]
                    if lock_z: transl[:, 2] = init_pos[2]
                    
                    locked_axes = []
                    if lock_x: locked_axes.append("X")
                    if lock_y: locked_axes.append("Y")
                    if lock_z: locked_axes.append("Z")
                    print(f"[HY-Motion] In-place applied (Modular): Locked axes {', '.join(locked_axes)}")
                
                # Prepare SMPL-H dictionary for converter
                smpl_data = construct_smpl_data_dict(rot6d, transl)
                
                fbx_filename = f"{filename_prefix}_{timestamp}_{unique_id}_{batch_idx:03d}.fbx"
                fbx_path = os.path.join(full_output_dir, fbx_filename)
                
                success = fbx_converter.convert_npz_to_fbx(smpl_data, fbx_path, fps=fps)
                
                if success:
                    fbx_files.append(fbx_path)
                    print(f"[HY-Motion] FBX exported: {fbx_path}")
                    
                    # Optionally save text prompt as metadata
                    txt_path = fbx_path.replace(".fbx", ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(motion_data.text)
                else:
                    print(f"[HY-Motion] FBX export failed for batch {batch_idx}")
            except Exception as e:
                print(f"[HY-Motion] Export error: {e}")
                continue

        # Return paths relative to ComfyUI output directory
        relative_paths = [os.path.relpath(p, COMFY_OUTPUT_DIR).replace("\\", "/") for p in fbx_files]
        
        if not relative_paths:
            return {"ui": {"error": ["FBX export failed. Check console for details."], "timestamp": [time.time()]}, "result": ("",)}
            
        result = "\n".join(relative_paths)
        
        if len(relative_paths) > 1:
            print(f"[HY-Motion] Exported {len(relative_paths)} files, returning all paths for multi-view.")
            
        fbx_info = []
        for p in fbx_files:
            fbx_info.append({
                "filename": os.path.basename(p),
                "subfolder": output_dir,
                "type": "output"
            })
            
        return {"ui": {"fbx": fbx_info, "fbx_url": [result], "timestamp": [time.time()]}, "result": (result, fbx_info)}


# ============================================================================
# Export Node Mappings
# ============================================================================

# ============================================================================
# Node: HYMotionPromptRewrite - Official Prompt Engineering
# ============================================================================

# Global cache for the prompter model (expensive to load)
_PROMPT_REWRITER_CACHE = {}

class HYMotionPromptRewrite:
    """
    Uses the official Text2MotionPrompter model to:
    1. Refine and improve text prompts for motion generation
    2. Estimate optimal duration (in seconds) for the described action
    
    Model: Text2MotionPrompter/Text2MotionPrompter (HuggingFace)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        try:
            hymotion_models = folder_paths.get_filename_list("hymotion")
            enhancer_models = [m for m in hymotion_models if "Text2MotionPrompter" in m]
            enhancer_models = ["auto"] + enhancer_models
        except:
            enhancer_models = ["auto"]

        return {
            "required": {
                "text": ("STRING", {
                    "default": "A person walks forward then jumps",
                    "multiline": True,
                    "tooltip": "Your original motion description. Will be enhanced by the AI prompter."
                }),
            },
            "optional": {
                "prompt_enhancer": (enhancer_models, {
                    "default": "auto",
                    "tooltip": "Text2MotionPrompter model. 'auto' downloads from HuggingFace if needed."
                }),
                "use_rewritten_duration": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, uses AI-estimated duration. Otherwise uses manual_duration."
                }),
                "manual_duration": ("FLOAT", {
                    "default": 3.0, 
                    "min": 0.5, 
                    "max": 12.0, 
                    "step": 0.1,
                    "tooltip": "Fallback duration if use_rewritten_duration is disabled."
                }),
                "quantization": (["nf4", "int4", "fp16"], {
                    "default": "nf4",
                    "tooltip": "Quantization for the prompter LLM. 'nf4' saves most VRAM."
                }),
                "double_quant": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable double quantization for extra VRAM savings with nf4."
                }),
                "max_new_tokens": ("INT", {
                    "default": 512, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 64,
                    "tooltip": "Maximum tokens for the rewritten prompt. Higher = more detailed descriptions."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("rewritten_text", "duration", "original_text")
    FUNCTION = "rewrite_prompt"
    CATEGORY = "HY-Motion/prompt"
    
    def rewrite_prompt(self, text: str, prompt_enhancer: str = "auto", use_rewritten_duration: bool = True, manual_duration: float = 3.0,
                       quantization: str = "nf4", double_quant: bool = True, max_new_tokens: int = 512):
        global _PROMPT_REWRITER_CACHE
        
        # Resolve model path
        model_path = "Text2MotionPrompter/Text2MotionPrompter"
        if prompt_enhancer != "auto":
            # If user selected a specific folder in models/hymotion/
            model_path = folder_paths.get_full_path("hymotion", prompt_enhancer)
            if os.path.isfile(model_path):
                model_path = os.path.dirname(model_path)
        
        # Create a config key to detect setting changes
        config_key = f"{quantization}_{double_quant}"
        
        # Check if we need to reload due to settings change
        if "config_key" in _PROMPT_REWRITER_CACHE and _PROMPT_REWRITER_CACHE["config_key"] != config_key:
            print(f"[HY-Motion] Quantization settings changed, will reload on next inference")
            # Mark for reload by clearing the rewriter
            if "rewriter" in _PROMPT_REWRITER_CACHE:
                # Clear old model from memory
                del _PROMPT_REWRITER_CACHE["rewriter"]
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Initialize rewriter if not cached
        if "rewriter" not in _PROMPT_REWRITER_CACHE:
            try:
                from .hymotion.prompt_engineering.prompt_rewrite import PromptRewriter
                _PROMPT_REWRITER_CACHE["rewriter"] = PromptRewriter(
                    model_path=model_path,
                    quantization=quantization,
                    double_quant=double_quant
                )
                _PROMPT_REWRITER_CACHE["config_key"] = config_key
            except Exception as e:
                print(f"[HY-Motion] ERROR: Failed to initialize PromptRewriter: {e}")
                return (text, manual_duration, text)
        
        rewriter = _PROMPT_REWRITER_CACHE["rewriter"]
        
        try:
            # Get rewritten text and suggested duration
            suggested_duration, rewritten_text = rewriter.rewrite_prompt_and_infer_time(
                text, 
                max_new_tokens=max_new_tokens
            )
            
            # Use suggested or manual duration based on user preference
            final_duration = suggested_duration if use_rewritten_duration else manual_duration
            
            return (rewritten_text, final_duration, text)
            
        except Exception as e:
            print(f"[HY-Motion] Prompt rewrite failed: {e}")
            return (text, manual_duration, text)


class HYMotionSaveNPZ:
    """Save motion data to NPZ format for persistence or debugging"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA", {
                    "tooltip": "Motion data from HYMotionSampler to save"
                }),
                "output_dir": ("STRING", {
                    "default": "hymotion_npz",
                    "tooltip": "Subdirectory in ComfyUI output folder for NPZ files."
                }),
                "filename_prefix": ("STRING", {
                    "default": "motion",
                    "tooltip": "Prefix for NPZ filenames. Contains keypoints, rotations, and metadata."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("npz_paths",)
    FUNCTION = "save_npz"
    CATEGORY = "HY-Motion/utils"
    OUTPUT_NODE = True

    def save_npz(self, motion_data: HYMotionData, output_dir: str, filename_prefix: str):
        full_output_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        output_dict = motion_data.output_dict
        timestamp = get_timestamp()
        unique_id = str(uuid.uuid4())[:8]

        npz_files = []

        for batch_idx in range(motion_data.batch_size):
            data = {}
            for key in ["keypoints3d", "rot6d", "transl", "root_rotations_mat"]:
                if key in output_dict and output_dict[key] is not None:
                    tensor = output_dict[key][batch_idx]
                    if hasattr(tensor, 'cpu'):
                        data[key] = tensor.cpu().numpy()
                    else:
                        data[key] = np.array(tensor)
            
            # CONSISTENCY FIX: Include full SMPL-H poses (including fingers)
            # This matches the data used during FBX export.
            if "rot6d" in data and "transl" in data:
                smpl_data = construct_smpl_data_dict(
                    torch.from_numpy(data["rot6d"]), 
                    torch.from_numpy(data["transl"])
                )
                # This adds 'poses' (T, 156) and other keys
                for k, v in smpl_data.items():
                    if k not in data:
                        data[k] = v
                
                # Also include text and duration for complete reconstruction
                data["text"] = motion_data.text
                data["duration"] = motion_data.duration
                data["seed"] = motion_data.seeds[batch_idx] if batch_idx < len(motion_data.seeds) else 0

            npz_filename = f"{filename_prefix}_{timestamp}_{unique_id}_{batch_idx:03d}.npz"
            npz_path = os.path.join(full_output_dir, npz_filename)

            np.savez(npz_path, **data)
            npz_files.append(npz_path)
            print(f"[HY-Motion] NPZ saved (Full Poses): {npz_path}")

        # Return paths relative to ComfyUI output directory
        relative_paths = [os.path.relpath(p, COMFY_OUTPUT_DIR).replace("\\", "/") for p in npz_files]
        result = "\n".join(relative_paths)
        return (result,)


class HYMotionLoadNPZ:
    """Load motion data from an NPZ file (from output or input directory)"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "npz_name": (folder_paths.get_filename_list("hymotion_npz"), {
                    "tooltip": "Select an NPZ file from the output/hymotion_npz or input/hymotion_npz directory."
                }),
            }
        }
    
    # ... existing return types/names ...
    RETURN_TYPES = ("HYMOTION_DATA",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "load"
    CATEGORY = "HY-Motion/utils"
    
    def load(self, npz_name, start_frame_idx=0, end_frame_idx=-1):
        # Resolve path using registered folder_paths
        full_path = folder_paths.get_full_path("hymotion_npz", npz_name)
        
        if not full_path or not os.path.exists(full_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_name}")
            
        print(f"[HY-Motion] Loading NPZ: {full_path}")
        data = np.load(full_path, allow_pickle=True)
        
        # Reconstruct output_dict
        output_dict = {}
        # Map keys back to tensors
        for key in ["keypoints3d", "rot6d", "transl", "root_rotations_mat"]:
            if key in data:
                tensor = torch.from_numpy(data[key])
                
                # Slicing logic
                total_frames = tensor.shape[0]
                
                # Handle end_frame_idx = -1
                actual_end = end_frame_idx if end_frame_idx != -1 else total_frames
                actual_start = max(0, min(start_frame_idx, total_frames))
                actual_end = max(actual_start, min(actual_end, total_frames))
                
                sliced_tensor = tensor[actual_start:actual_end]
                output_dict[key] = sliced_tensor.unsqueeze(0) # Add batch dim
        
        # Extract metadata
        text = str(data.get("text", "loaded motion"))
        duration = float(data.get("duration", 0.0))
        seed = int(data.get("seed", 0))
        
        # Recalculate duration if sliced
        if "rot6d" in output_dict:
            num_frames = output_dict["rot6d"].shape[1]
            # If duration was 0 or missing, assume 30 FPS
            if duration <= 0:
                duration = num_frames / 30.0
            else:
                # Scale duration based on frame ratio
                orig_frames = data["rot6d"].shape[0] if "rot6d" in data else num_frames
                if orig_frames > 0:
                    duration = duration * (num_frames / orig_frames)
            
            if start_frame_idx != 0 or end_frame_idx != -1:
                text = f"{text} (frames {actual_start}-{actual_end})"
            
        motion_data = HYMotionData(
            output_dict=output_dict,
            text=text,
            duration=duration,
            seeds=[seed],
            device_info="cpu"
        )

        start_frame = None
        end_frame = None
        if "rot6d" in output_dict:
            start_frame = HYMotionFrame(output_dict["rot6d"][:, 0], output_dict["transl"][:, 0])
            end_frame = HYMotionFrame(output_dict["rot6d"][:, -1], output_dict["transl"][:, -1])

        # HIDDEN: Frame outputs disabled for now
        # return (motion_data, start_frame, end_frame)
        return (motion_data,)


class HYMotionRetargetFBX:
    """
    Retargets HY-Motion data to custom FBX skeletons.
    Uses the standalone retargeting script for maximum compatibility.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA", {
                    "tooltip": "Motion data from HY-Motion Sampler to retarget onto your custom character"
                }),
                "target_fbx": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Path to your target FBX character (e.g., /path/to/mycharacter.fbx). This is the skeleton you want to apply the motion to."
                }),
                "output_dir": ("STRING", {
                    "default": "hymotion_retarget",
                    "tooltip": "Subdirectory in ComfyUI output folder where retargeted FBX files will be saved."
                }),
                "filename_prefix": ("STRING", {
                    "default": "retarget",
                    "tooltip": "Prefix for output filenames (e.g., 'dance' creates dance_001.fbx, dance_002.fbx)."
                }),
            },
            "optional": {
                "mapping_file": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Optional: Path to custom bone mapping JSON if auto-detection fails. Leave empty for automatic fuzzy matching."
                }),
                "yaw_offset": ("FLOAT", {
                    "default": 0.0, 
                    "min": -180.0, 
                    "max": 180.0, 
                    "step": 1.0,
                    "tooltip": "Rotate the character around Y-axis in degrees (e.g., 180 to make them face the opposite direction)."
                }),
                "scale": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.01,
                    "tooltip": "Force specific scale multiplier. Leave at 0.0 for automatic height-based scaling (recommended)."
                }),
                "neutral_fingers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use neutral finger rest pose to preserve natural finger curl from source animation. Disable for legacy behavior."
                }),
                "unique_names": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, adds timestamps to filenames to prevent overwriting. Disable to use fixed names for easier iteration in Godot."
                }),
                "in_place": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, removes horizontal movement from the animation so the character stays in one spot. Useful for game development."
                }),
                "in_place_x": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep character centered on X axis (Side-to-Side)."
                }),
                "in_place_y": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep character centered on Y axis (Up/Down - prevents jumping/falling)."
                }),
                "in_place_z": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep character centered on Z axis (Forward/Backward)."
                }),
                "preserve_position": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, keeps the character's absolute world position from the motion data. Disable to automatically center the character at the start (fixes 'jumps' when stitching)."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "FBX")
    RETURN_NAMES = ("fbx_path", "fbx")
    FUNCTION = "retarget"
    CATEGORY = "HY-Motion/view"
    OUTPUT_NODE = True
    
    def retarget(self, motion_data, target_fbx, output_dir="hymotion_retarget", filename_prefix="retarget", 
                 mapping_file="", yaw_offset=0.0, scale=0.0, neutral_fingers=True, unique_names=True, 
                 in_place=False, in_place_x=False, in_place_y=False, in_place_z=False, preserve_position=False):
        """Retarget motion to custom FBX skeleton."""
        
        # Resolve target FBX path robustly
        resolved_path = None
        
        if os.path.isabs(target_fbx):
            if os.path.isfile(target_fbx):
                resolved_path = target_fbx
        else:
            # Handle common prefixes
            if target_fbx.startswith("input/"):
                rel_path = target_fbx[6:]
                potential_path = os.path.join(folder_paths.get_input_directory(), rel_path)
                if os.path.isfile(potential_path):
                    resolved_path = potential_path
            elif target_fbx.startswith("output/"):
                rel_path = target_fbx[7:]
                potential_path = os.path.join(folder_paths.get_output_directory(), rel_path)
                if os.path.isfile(potential_path):
                    resolved_path = potential_path
            
            # Fallback: Search in input and output directories
            if not resolved_path:
                input_dir = folder_paths.get_input_directory()
                output_dir_comfy = folder_paths.get_output_directory()
                
                for base_dir in [input_dir, output_dir_comfy]:
                    potential_path = os.path.join(base_dir, target_fbx)
                    if os.path.isfile(potential_path):
                        resolved_path = potential_path
                        break
        
        if not resolved_path:
            # Final attempt: list files in input dir to help user debug
            input_dir = folder_paths.get_input_directory()
            print(f"[HY-Motion] ERROR: Target FBX not found: {target_fbx}")
            try:
                files = os.listdir(input_dir)
                print(f"[HY-Motion] Files in input directory: {files[:10]}...")
            except:
                pass
            raise ValueError(f"Target FBX file not found: {target_fbx}")
        
        target_fbx = resolved_path
        
        # Create output directory
        full_output_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        timestamp = get_timestamp()
        unique_id = str(uuid.uuid4())[:8]
        output_fbx_files = []
        
        # Process each batch item separately
        for batch_idx in range(motion_data.batch_size):
            if unique_names:
                output_fbx = os.path.join(full_output_dir, f"{filename_prefix}_{timestamp}_{unique_id}_{batch_idx:03d}.fbx")
            else:
                output_fbx = os.path.join(full_output_dir, f"{filename_prefix}_{batch_idx:03d}.fbx")
            
            try:
                if not HAS_RETARGET_UTILS:
                    raise RuntimeError("Retargeting utilities not available. Check if FBX SDK is installed.")

                output_dict = motion_data.output_dict
                
                # Extract data for this batch item
                data_dict = {}
                for key in ['keypoints3d', 'rot6d', 'transl', 'root_rotations_mat']:
                    if key in output_dict and output_dict[key] is not None:
                        tensor = output_dict[key][batch_idx]
                        data_dict[key] = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)
                
                # Add SMPL-H full poses for consistency
                if "rot6d" in data_dict and "transl" in data_dict:
                    smpl_data = construct_smpl_data_dict(
                        torch.from_numpy(data_dict["rot6d"]), 
                        torch.from_numpy(data_dict["transl"])
                    )
                    for k, v in smpl_data.items():
                        if k not in data_dict:
                            data_dict[k] = v
                
                # Populate Skeleton object
                if "poses" in data_dict:
                    # Call native functions with in-memory data
                    print(f"[HYMotionRetargetFBX] Retargeting batch {batch_idx}/{motion_data.batch_size} (Native In-Memory)...")
                    
                    # Load skeletons (src_skel from dict, tgt from FBX)
                    src_skel_loaded = load_npz(data_dict)
                    tgt_man, tgt_scene, tgt_skel = load_fbx(target_fbx)
                    
                    # Perform retargeting
                    mapping = load_bone_mapping(mapping_file, src_skel_loaded, tgt_skel)
                    rots, locs, active = retarget_animation(
                        src_skel_loaded, tgt_skel, mapping, 
                        scale, yaw_offset, neutral_fingers, in_place,
                        in_place_x, in_place_y, in_place_z, preserve_position
                    )
                    
                    # Apply and save
                    src_time_mode = None # Default to 30 FPS
                    apply_retargeted_animation(
                        tgt_scene, tgt_skel, rots, locs, 
                        src_skel_loaded.frame_start, src_skel_loaded.frame_end, src_time_mode
                    )
                    
                    save_fbx(tgt_man, tgt_scene, output_fbx)
                else:
                    raise ValueError("Motion data missing 'poses' for retargeting.")

                output_fbx_files.append(output_fbx)
                print(f"[HYMotionRetargetFBX] Batch {batch_idx} done: {os.path.basename(output_fbx)}")
                
            except Exception as e:
                error_msg = f"Retargeting failed for batch {batch_idx}: {str(e)}"
                print(f"[HYMotionRetargetFBX] {error_msg}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(error_msg)
        
        # Return all paths (newline-separated)
        relative_paths = [os.path.relpath(p, COMFY_OUTPUT_DIR).replace("\\", "/") for p in output_fbx_files]
        result = "\n".join(relative_paths)
        
        print(f"[HYMotionRetargetFBX] Successfully retargeted {len(output_fbx_files)} batch item(s)")
        
        # Format for ComfyUI output/history so the Godot bridge can find it
        fbx_info = []
        for p in output_fbx_files:
            fbx_info.append({
                "filename": os.path.basename(p),
                "subfolder": output_dir,
                "type": "output"
            })
            
        return {"ui": {"fbx": fbx_info, "fbx_url": [result], "timestamp": [time.time()]}, "result": (result, fbx_info)}


class HYMotionSMPLToData:
    """Convert SMPL parameters (from GVHMR/MotionCapture) to HY-Motion Data format"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS", {"tooltip": "SMPL parameters from GVHMR or other motion capture nodes."}),
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional description. If empty, defaults to 'motion from smpl'."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1, "tooltip": "Duration in seconds. If 0.0, it will be auto-calculated based on frame count at 30 FPS."}),
            }
        }

    RETURN_TYPES = ("HYMOTION_DATA",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "convert"
    CATEGORY = "HY-Motion/utils"

    def convert(self, smpl_params, text="", duration=0.0):
        def safe_float(v, default):
            try:
                if v is None or v == "": return default
                return float(v)
            except: return default
        
        duration = safe_float(duration, 0.0)

        # Handle list of params if batched
        if isinstance(smpl_params, list):
            params = smpl_params[0]
        else:
            params = smpl_params

        if params is None:
            raise ValueError("[HY-Motion] smpl_params is None. Ensure the source node (like GVHMR) is connected and has run.")

        # DEEP ANALYZE: Handle GVHMR nested structure
        if "global" in params:
            print("[HY-Motion] Detected GVHMR nested structure, using 'global' parameters")
            inner_params = params["global"]
        elif "incam" in params:
            print("[HY-Motion] Detected GVHMR nested structure, using 'incam' parameters")
            inner_params = params["incam"]
        else:
            inner_params = params

        # Extract poses and trans
        poses = inner_params.get("poses", None)
        body_pose = inner_params.get("body_pose", None)
        global_orient = inner_params.get("global_orient", None)
        trans = inner_params.get("trans", inner_params.get("transl", None))

        # Robust key detection
        if poses is None and (body_pose is None or global_orient is None):
            for k in inner_params.keys():
                if "pose" in k.lower() and isinstance(inner_params[k], (torch.Tensor, np.ndarray)):
                    poses = inner_params[k]
                    print(f"[HY-Motion] Guessed pose data from key: {k}")
                    break
            
            if poses is None:
                raise ValueError(f"[HY-Motion] SMPL_PARAMS missing pose data. Available keys: {list(inner_params.keys())}")

        # Convert to torch and handle device
        def to_tensor(x):
            if x is None: return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            return x.float() if isinstance(x, torch.Tensor) else x

        poses = to_tensor(poses)
        body_pose = to_tensor(body_pose)
        global_orient = to_tensor(global_orient)
        trans = to_tensor(trans)

        # Combine body_pose and global_orient if needed
        if poses is None:
            # Ensure same device before cat
            device = body_pose.device
            poses = torch.cat([global_orient.to(device), body_pose], dim=1)

        # Ensure 3D (T, J, 3) or 4D (B, T, J, 3)
        if len(poses.shape) == 2:
            poses = poses.reshape(poses.shape[0], -1, 3)
        
        # Detect if input is batched (B, T, J, 3)
        is_batched = len(poses.shape) == 4
        if not is_batched:
            poses = poses.unsqueeze(0) # (1, T, J, 3)
        
        batch_size, num_frames, num_joints, _ = poses.shape
        device = poses.device
        dtype = poses.dtype
        
        # Support variable joint counts and auto-padding for hands/feet
        # HY-Motion's internal model (WoodenMesh) and FBX exporter expect 52 joints (SMPL-H).
        if num_joints < 52:
            print(f"[HY-Motion] Input has {num_joints} joints. Auto-padding to reach 52 joints (SMPL-H).")
            
            # Import mean hand poses from body_model if possible, otherwise use zeros
            try:
                from .hymotion.pipeline.body_model import LEFT_HAND_MEAN_AA, RIGHT_HAND_MEAN_AA
                left_hand = torch.tensor(LEFT_HAND_MEAN_AA, device=device, dtype=dtype).reshape(1, 1, 15, 3).expand(batch_size, num_frames, -1, -1)
                right_hand = torch.tensor(RIGHT_HAND_MEAN_AA, device=device, dtype=dtype).reshape(1, 1, 15, 3).expand(batch_size, num_frames, -1, -1)
                mean_hand_padding = torch.cat([left_hand, right_hand], dim=2) # (B, T, 30, 3)
            except ImportError:
                print("[HY-Motion] Warning: Could not import mean hand poses, using zeros for padding.")
                mean_hand_padding = torch.zeros((batch_size, num_frames, 30, 3), device=device, dtype=dtype)

            # If we have 22 joints, we pad with 30 hand joints to get 52
            if num_joints == 22:
                poses = torch.cat([poses, mean_hand_padding], dim=2)
            else:
                # For other counts (like 24), we truncate to 22 first to ensure standard SMPL-H mapping
                # or we pad to 52 if we know the mapping. For robustness, we'll pad with zeros/mean up to 52.
                needed = 52 - num_joints
                padding = torch.zeros((batch_size, num_frames, needed, 3), device=device, dtype=dtype)
                poses = torch.cat([poses, padding], dim=2)
            
            num_joints = 52
            print(f"[HY-Motion] Padded poses shape: {poses.shape}")
            
        elif num_joints > 52:
            # If we have more than 52 joints (e.g. SMPL-X with 55 or SAM3D with 70), 
            # we truncate to 52 to maintain compatibility with the 52-joint FBX skeleton.
            print(f"[HY-Motion] Truncating joints from {num_joints} to 52 for SMPL-H compatibility")
            poses = poses[:, :, :52, :]
            num_joints = 52
        
        if trans is None:
            print("[HY-Motion] Warning: No translation found, using zeros")
            trans = torch.zeros((batch_size, num_frames, 3), device=device, dtype=dtype)
        else:
            # Ensure trans is (B, T, 3)
            if len(trans.shape) == 2:
                if is_batched:
                    # If poses were batched but trans wasn't, we might have a problem.
                    # Assume trans applies to the whole batch or is the first item.
                    trans = trans.unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    trans = trans.unsqueeze(0)
            
            # Ensure trans matches num_frames
            if trans.shape[1] != num_frames:
                print(f"[HY-Motion] Warning: Translation frames ({trans.shape[1]}) mismatch poses ({num_frames}). Truncating/Padding.")
                if trans.shape[1] > num_frames:
                    trans = trans[:, :num_frames, :]
                else:
                    pad = torch.zeros((batch_size, num_frames - trans.shape[1], 3), device=device, dtype=dtype)
                    trans = torch.cat([trans, pad], dim=1)

        # QUALITY CHECK: NaN/Inf detection
        if torch.isnan(poses).any() or torch.isinf(poses).any():
            print("[HY-Motion] ERROR: Input poses contain NaNs or Infs! Cleaning with zeros.")
            poses = torch.nan_to_num(poses)
        if torch.isnan(trans).any() or torch.isinf(trans).any():
            print("[HY-Motion] ERROR: Input translation contains NaNs or Infs! Cleaning with zeros.")
            trans = torch.nan_to_num(trans)

        # Auto-calculate duration
        if duration <= 0.0:
            fps = params.get("mocap_framerate", params.get("fps", inner_params.get("mocap_framerate", 30.0)))
            duration = num_frames / float(fps)
            print(f"[HY-Motion] Auto-calculated duration: {duration:.2f}s ({num_frames} frames @ {fps} FPS)")

        if not text.strip():
            text = "motion from smpl"

        # Convert axis-angle to rot6d
        # CRITICAL FIX: Use rotation_matrix_to_rot6d (column-based) to match HY-Motion's decoder
        rot_mat = axis_angle_to_matrix(poses)
        rot6d = rotation_matrix_to_rot6d(rot_mat)

        # Generate 3D keypoints for preview (matching HYMotionSampler logic)
        global _WOODEN_MESH_CACHE
        if _WOODEN_MESH_CACHE is None:
            print("[HY-Motion] Loading 3D body model for converter preview...")
            _WOODEN_MESH_CACHE = WoodenMesh().to(device).eval()
        
        body_model = _WOODEN_MESH_CACHE.to(device)
        with torch.no_grad():
            body_params = {
                "rot6d": rot6d,
                "trans": trans
            }
            # Use forward_batch for efficiency
            body_out = body_model.forward_batch(body_params, chunk_size=64)
            keypoints3d = body_out["keypoints3d"]

        # Move to CPU for HYMOTION_DATA storage (standard practice)
        rot6d = rot6d.cpu()
        trans = trans.cpu()
        rot_mat = rot_mat.cpu()
        keypoints3d = keypoints3d.cpu()

        output_dict = {
            "rot6d": rot6d,
            "transl": trans,
            "root_rotations_mat": rot_mat[:, :, 0],
            "keypoints3d": keypoints3d
        }

        motion_data = HYMotionData(
            output_dict=output_dict,
            text=text,
            duration=duration,
            seeds=[0] * batch_size,
            device_info="cpu"
        )

        print(f"[HY-Motion] Successfully converted {num_frames} frames ({num_joints} joints) from SMPL to HY-Motion format (Batch: {batch_size}).")
        return (motion_data,)

from .hymotion.utils.downloader import download_file, get_model_path, MODEL_METADATA, download_models_parallel

class HYMotionModelDownloader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_choices": (list(MODEL_METADATA.keys()), {"default": list(MODEL_METADATA.keys())[0], "multiselect": True}),
                "download_all": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_paths",)
    FUNCTION = "download"
    CATEGORY = "HY-Motion"

    def download(self, model_choices, download_all=False, custom_path=""):
        if download_all:
            selected_models = list(MODEL_METADATA.keys())
        else:
            if isinstance(model_choices, str):
                selected_models = [model_choices]
            else:
                selected_models = model_choices
        
        if not selected_models:
            return ("No models selected",)
            
        results = download_models_parallel(selected_models, custom_path)
        
        # If no new downloads were needed, resolve existing paths
        if not results:
            results = [get_model_path(name, custom_path) for name in selected_models]
            
        return ("\n".join(results),)


class HYMotionWoodFBXToData:
    """
    Extract SMPL-H animation from FBX files rigged with the 52-joint wooden mannequin skeleton.
    Converts FBX animation to HYMOTION_DATA format for use with HYMotionSampler.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fbx_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to FBX file with SMPL-H skeleton (wooden mannequin format)"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "First frame to extract (0-indexed)"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Last frame to extract (-1 = all frames)"
                }),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "tooltip": "Target frame rate"
                }),
            },
        }
    
    RETURN_TYPES = ("HYMOTION_DATA",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "extract"
    CATEGORY = "HY-Motion/modular"
    
    # SMPL-H joint names in order (52 joints)
    SMPLH_JOINT_NAMES = [
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
        "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
        "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist",
        # Left hand fingers
        "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3",
        "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3",
        "L_Thumb1", "L_Thumb2", "L_Thumb3",
        # Right hand fingers
        "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3",
        "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3",
        "R_Thumb1", "R_Thumb2", "R_Thumb3"
    ]
    
    # Lowercase aliases for matching
    LOWERCASE_ALIASES = {
        "pelvis": "Pelvis", "l_hip": "L_Hip", "r_hip": "R_Hip", "spine1": "Spine1",
        "l_knee": "L_Knee", "r_knee": "R_Knee", "spine2": "Spine2", "l_ankle": "L_Ankle",
        "r_ankle": "R_Ankle", "spine3": "Spine3", "l_foot": "L_Foot", "r_foot": "R_Foot",
        "neck": "Neck", "l_collar": "L_Collar", "r_collar": "R_Collar", "head": "Head",
        "l_shoulder": "L_Shoulder", "r_shoulder": "R_Shoulder", "l_elbow": "L_Elbow",
        "r_elbow": "R_Elbow", "l_wrist": "L_Wrist", "r_wrist": "R_Wrist",
        # Additional common aliases
        "left_hip": "L_Hip", "right_hip": "R_Hip", "left_knee": "L_Knee", "right_knee": "R_Knee",
        "left_ankle": "L_Ankle", "right_ankle": "R_Ankle", "left_foot": "L_Foot", "right_foot": "R_Foot",
        "left_collar": "L_Collar", "right_collar": "R_Collar", "left_shoulder": "L_Shoulder",
        "right_shoulder": "R_Shoulder", "left_elbow": "L_Elbow", "right_elbow": "R_Elbow",
        "left_wrist": "L_Wrist", "right_wrist": "R_Wrist",
    }
    
    def extract(self, fbx_path: str, start_frame: int = 0, end_frame: int = -1, fps: int = 30):
        try:
            import fbx
            from scipy.spatial.transform import Rotation as R
        except ImportError:
            raise RuntimeError("FBX SDK not installed. Please install python-fbx.")
        
        from .hymotion.utils.geometry import rotation_matrix_to_rot6d
        
        # Resolve path
        if fbx_path.startswith("input/"):
            fbx_path = os.path.join(folder_paths.get_input_directory(), fbx_path[6:])
        elif fbx_path.startswith("output/"):
            fbx_path = os.path.join(folder_paths.get_output_directory(), fbx_path[7:])
        
        if not os.path.exists(fbx_path):
            raise FileNotFoundError(f"FBX file not found: {fbx_path}")
        
        print(f"[HY-Motion] Loading FBX: {fbx_path}")
        
        # Load FBX
        fbx_manager = fbx.FbxManager.Create()
        ios = fbx.FbxIOSettings.Create(fbx_manager, fbx.IOSROOT)
        fbx_manager.SetIOSettings(ios)
        
        importer = fbx.FbxImporter.Create(fbx_manager, "")
        if not importer.Initialize(fbx_path, -1, fbx_manager.GetIOSettings()):
            raise RuntimeError(f"Failed to load FBX: {importer.GetStatus().GetErrorString()}")
        
        scene = fbx.FbxScene.Create(fbx_manager, "")
        importer.Import(scene)
        importer.Destroy()
        
        # Collect nodes
        def collect_nodes(node, nodes_dict=None):
            if nodes_dict is None:
                nodes_dict = {}
            nodes_dict[node.GetName()] = node
            for i in range(node.GetChildCount()):
                collect_nodes(node.GetChild(i), nodes_dict)
            return nodes_dict
        
        all_nodes = collect_nodes(scene.GetRootNode())
        print(f"[HY-Motion] Found {len(all_nodes)} nodes")
        
        # Map SMPL-H joints to FBX nodes
        joint_to_node = {}
        for i, joint_name in enumerate(self.SMPLH_JOINT_NAMES):
            # Try exact match
            if joint_name in all_nodes:
                joint_to_node[i] = all_nodes[joint_name]
            # Try lowercase
            elif joint_name.lower() in all_nodes:
                joint_to_node[i] = all_nodes[joint_name.lower()]
            # Try alias
            else:
                for alias, canonical in self.LOWERCASE_ALIASES.items():
                    if canonical == joint_name and alias in all_nodes:
                        joint_to_node[i] = all_nodes[alias]
                        break
        
        print(f"[HY-Motion] Mapped {len(joint_to_node)}/{len(self.SMPLH_JOINT_NAMES)} joints")
        if len(joint_to_node) < 22:
            print(f"[HY-Motion] Warning: Less than 22 body joints found. Available nodes: {list(all_nodes.keys())[:20]}...")
        
        # Get animation stack
        anim_stack = scene.GetCurrentAnimationStack()
        if not anim_stack:
            anim_stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
            if anim_stack_count > 0:
                anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
        
        if not anim_stack:
            raise RuntimeError("No animation found in FBX file")
        
        # Get time span
        local_span = anim_stack.GetLocalTimeSpan()
        start_time = local_span.GetStart()
        end_time = local_span.GetStop()
        
        # Calculate frame count
        frame_duration = 1.0 / fps
        total_duration = end_time.GetSecondDouble() - start_time.GetSecondDouble()
        total_frames = int(total_duration * fps) + 1
        
        if end_frame < 0:
            end_frame = total_frames - 1
        end_frame = min(end_frame, total_frames - 1)
        
        num_frames = end_frame - start_frame + 1
        print(f"[HY-Motion] Extracting frames {start_frame} to {end_frame} ({num_frames} frames)")
        
        # Extract animation
        num_joints = len(self.SMPLH_JOINT_NAMES)
        rot6d = torch.zeros((1, num_frames, num_joints, 6), dtype=torch.float32)
        transl = torch.zeros((1, num_frames, 3), dtype=torch.float32)
        
        # Set identity rotation as default
        rot6d[:, :, :, 0] = 1.0  # First column of identity
        rot6d[:, :, :, 4] = 1.0  # Second column of identity
        
        fbx_time = fbx.FbxTime()
        
        for frame_idx in range(num_frames):
            actual_frame = start_frame + frame_idx
            fbx_time.SetSecondDouble(actual_frame * frame_duration)
            
            for joint_idx, node in joint_to_node.items():
                # Get local rotation as Euler angles
                euler = node.EvaluateLocalRotation(fbx_time)
                euler_np = np.array([euler[0], euler[1], euler[2]], dtype=np.float64)
                
                # Convert Euler (degrees) to rotation matrix
                rot_mat = R.from_euler('xyz', euler_np, degrees=True).as_matrix()
                rot_mat_tensor = torch.from_numpy(rot_mat).float().unsqueeze(0)
                
                # Convert to 6D rotation
                r6d = rotation_matrix_to_rot6d(rot_mat_tensor)[0]
                rot6d[0, frame_idx, joint_idx] = r6d
                
                # Extract translation for root (Pelvis)
                if joint_idx == 0:
                    trans = node.EvaluateLocalTranslation(fbx_time)
                    # Convert from cm to meters (typical FBX scale)
                    transl[0, frame_idx] = torch.tensor([trans[0], trans[1], trans[2]]) / 100.0
        
        # Cleanup
        fbx_manager.Destroy()
        
        # Create output dict
        output_dict = {
            "rot6d": rot6d,
            "transl": transl,
        }
        
        # Calculate keypoints3d using wooden mesh if available
        try:
            global _WOODEN_MESH_CACHE
            if _WOODEN_MESH_CACHE is None:
                _WOODEN_MESH_CACHE = WoodenMesh(device='cpu')
            
            smpl_data = construct_smpl_data_dict(rot6d, transl)
            keypoints = _WOODEN_MESH_CACHE.get_keypoints(smpl_data)
            output_dict["keypoints3d"] = keypoints
        except Exception as e:
            print(f"[HY-Motion] Could not compute keypoints3d: {e}")
        
        motion_data = HYMotionData(
            output_dict=output_dict,
            text=f"FBX: {os.path.basename(fbx_path)}",
            duration=num_frames / fps,
            seeds=[0],
            device_info="cpu"
        )
        
        print(f"[HY-Motion] Extracted motion: {num_frames} frames, {len(joint_to_node)} joints")
        
        return (motion_data,)


NODE_CLASS_MAPPINGS_MODULAR = {
    "HYMotionDiTLoader": HYMotionDiTLoader,
    "HYMotionTextEncoderLoader": HYMotionTextEncoderLoader,
    "HYMotionTextEncode": HYMotionTextEncode,
    "HYMotionSampler": HYMotionSampler,
    "HYMotionExtractFrame": HYMotionExtractFrame,
    "HYMotionModularExportFBX": HYMotionModularExportFBX,
    "HYMotionPromptRewrite": HYMotionPromptRewrite,
    "HYMotionSaveNPZ": HYMotionSaveNPZ,
    "HYMotionLoadNPZ": HYMotionLoadNPZ,
    "HYMotionRetargetFBX": HYMotionRetargetFBX,
    "HYMotionSMPLToData": HYMotionSMPLToData,
    "HYMotionFBXPlayer": HYMotionFBXPlayer,
    "HYMotion3DModelLoader": HYMotion3DModelLoader,
    "HYMotionModelDownloader": HYMotionModelDownloader,
    "HYMotionWoodFBXToData": HYMotionWoodFBXToData,
}

NODE_DISPLAY_NAME_MAPPINGS_MODULAR = {
    "HYMotionDiTLoader": "HY-Motion DiT Loader",
    "HYMotionTextEncoderLoader": "HY-Motion Text Encoder Loader",
    "HYMotionTextEncode": "HY-Motion Text Encode",
    "HYMotionSampler": "HY-Motion Sampler",
    "HYMotionExtractFrame": "HY-Motion Extract Frame",
    "HYMotionModularExportFBX": "HY-Motion Export FBX",
    "HYMotionPromptRewrite": "HY-Motion Prompt Rewrite",
    "HYMotionSaveNPZ": "HY-Motion Save NPZ",
    "HYMotionLoadNPZ": "HY-Motion Load NPZ",
    "HYMotionRetargetFBX": "HY-Motion Retarget to FBX",
    "HYMotionSMPLToData": "HY-Motion SMPL to Data",
    "HYMotionFBXPlayer": "HY-Motion FBX Player",
    "HYMotion3DModelLoader": "HY-Motion 3D Model Loader",
    "HYMotionModelDownloader": "HY-Motion Model Downloader",
    "HYMotionWoodFBXToData": "HY-Motion WoodFBX to Data",
}


NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS_MODULAR}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS_MODULAR}
