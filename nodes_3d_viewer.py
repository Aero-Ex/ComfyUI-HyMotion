import os
import json
import torch
import numpy as np
import folder_paths
from server import PromptServer
from aiohttp import web

class HYMotion3DModelLoader:
    """
    Standalone viewer for 3D models (FBX, GLB, GLTF, OBJ).
    Scans both input and output directories.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        
        files = []
        extensions = [".fbx", ".glb", ".gltf", ".obj"]
        
        def scan_dir(base_path, prefix=""):
            if not os.path.exists(base_path): return
            for root, _, filenames in os.walk(base_path):
                for f in filenames:
                    if any(f.lower().endswith(ext) for ext in extensions):
                        rel = os.path.relpath(os.path.join(root, f), base_path).replace("\\", "/")
                        files.append(f"{prefix}{rel}")

        scan_dir(input_dir, "input/")
        scan_dir(output_dir, "output/")
        
        if not files: files = ["none"]
        else: files = sorted(files)

        return {
            "required": {
                "model_path": (files, {
                    "default": files[0],
                    "tooltip": "Select a 3D model to view immediately"
                }),
                "translate_x": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "translate_y": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "translate_z": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "rotate_x": ("FLOAT", {"default": 0.0, "step": 0.1}),
                "rotate_y": ("FLOAT", {"default": 0.0, "step": 0.1}),
                "rotate_z": ("FLOAT", {"default": 0.0, "step": 0.1}),
                "scale_x": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "scale_y": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "scale_z": ("FLOAT", {"default": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_url",)
    FUNCTION = "load_model"
    CATEGORY = "HY-Motion/view"
    OUTPUT_NODE = True

    def load_model(self, model_path, translate_x=0, translate_y=0, translate_z=0, rotate_x=0, rotate_y=0, rotate_z=0, scale_x=1, scale_y=1, scale_z=1):
        if model_path == "none":
            return (None,)
        
        ext = os.path.splitext(model_path)[1].lower()[1:]
        return {
            "ui": {
                "model_url": model_path, 
                "format": ext,
                "transform": {
                    "translate": [translate_x, translate_y, translate_z],
                    "rotate": [rotate_x, rotate_y, rotate_z],
                    "scale": [scale_x, scale_y, scale_z]
                }
            }, 
            "result": (model_path,)
        }

class HYMotionFBXPlayer:
    """
    (Legacy) Specific player for FBX files in the output directory.
    Useful for existing workflows or dedicated FBX playback.
    """
    @classmethod
    def INPUT_TYPES(s):
        output_dir = folder_paths.get_output_directory()
        fbx_files = []
        if os.path.exists(output_dir):
            for root, _, files in os.walk(output_dir):
                for f in files:
                    if f.lower().endswith(".fbx"):
                        rel = os.path.relpath(os.path.join(root, f), output_dir).replace("\\", "/")
                        fbx_files.append(rel)
        fbx_files = ["none"] + sorted(fbx_files)

        return {
            "required": {
                "fbx_name": (fbx_files, {"default": fbx_files[0]}),
            },
            "optional": {
                "fbx_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "play_fbx"
    CATEGORY = "HY-Motion/view"
    OUTPUT_NODE = True

    def play_fbx(self, fbx_name, fbx_path=""):
        selected = fbx_path.strip() if fbx_path and fbx_path.strip() else fbx_name
        if selected == "none": return (None,)
        # Map to "output/" prefix for consistency with the generic loader's logic
        full_path = f"output/{selected}".replace("\\", "/") if not selected.startswith("output/") else selected.replace("\\", "/")
        return {"ui": {"fbx_url": full_path}, "result": (full_path,)}

NODE_CLASS_MAPPINGS = {
    "HYMotionFBXPlayer": HYMotionFBXPlayer,
    "HYMotion3DModelLoader": HYMotion3DModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionFBXPlayer": "HY-Motion 3D FBX Player (Legacy)",
    "HYMotion3DModelLoader": "HY-Motion 3D Model Loader"
}
