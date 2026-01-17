import os
import json
import torch
import numpy as np
import folder_paths
from server import PromptServer
from aiohttp import web

# Server route for in-place FBX export
@PromptServer.instance.routes.post("/hymotion/export_inplace")
async def export_inplace_fbx(request):
    """
    Apply in-place animation to an FBX file (zeroes X/Z root translation).
    Request JSON: { "input_path": "output/path/to/file.fbx" }
    Returns the processed FBX file.
    """
    try:
        data = await request.json()
        input_path = data.get("input_path", "")
        
        if not input_path:
            return web.json_response({"error": "No input_path provided"}, status=400)
        
        # Resolve the full path
        if input_path.startswith("output/"):
            rel_path = input_path[7:]  # Remove "output/" prefix
            full_path = os.path.join(folder_paths.get_output_directory(), rel_path)
        elif input_path.startswith("input/"):
            rel_path = input_path[6:]  # Remove "input/" prefix
            full_path = os.path.join(folder_paths.get_input_directory(), rel_path)
        else:
            full_path = input_path
        
        if not os.path.exists(full_path):
            return web.json_response({"error": f"File not found: {full_path}"}, status=404)
        
        # Try to import FBX SDK and apply in-place
        try:
            import fbx
            from fbx import FbxTime, FbxAnimStack, FbxAnimLayer
            
            # Load the FBX
            manager = fbx.FbxManager.Create()
            ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
            manager.SetIOSettings(ios)
            
            importer = fbx.FbxImporter.Create(manager, "")
            if not importer.Initialize(full_path, -1, ios):
                manager.Destroy()
                return web.json_response({"error": "Failed to load FBX"}, status=500)
            
            scene = fbx.FbxScene.Create(manager, "")
            importer.Import(scene)
            importer.Destroy()
            
            # Find the root bone (Pelvis, Hips, etc)
            root_names = ["Pelvis", "pelvis", "Hips", "hips", "Root", "root", "mixamorig:Hips"]
            root_node = None
            
            def find_node(node, names):
                if node.GetName() in names:
                    return node
                for i in range(node.GetChildCount()):
                    result = find_node(node.GetChild(i), names)
                    if result:
                        return result
                return None
            
            root_node = find_node(scene.GetRootNode(), root_names)
            
            if root_node:
                # Get animation stack
                stack = scene.GetCurrentAnimationStack()
                if stack:
                    layer = stack.GetMember(0) if stack.GetMemberCount() > 0 else None
                    if layer:
                        # Get X and Z translation curves
                        tx_curve = root_node.LclTranslation.GetCurve(layer, "X", False)
                        tz_curve = root_node.LclTranslation.GetCurve(layer, "Z", False)
                        
                        # Get initial values at frame 0
                        if tx_curve and tx_curve.KeyGetCount() > 0:
                            init_x = tx_curve.KeyGetValue(0)
                            tx_curve.KeyModifyBegin()
                            for i in range(tx_curve.KeyGetCount()):
                                tx_curve.KeySetValue(i, init_x)
                            tx_curve.KeyModifyEnd()
                        
                        if tz_curve and tz_curve.KeyGetCount() > 0:
                            init_z = tz_curve.KeyGetValue(0)
                            tz_curve.KeyModifyBegin()
                            for i in range(tz_curve.KeyGetCount()):
                                tz_curve.KeySetValue(i, init_z)
                            tz_curve.KeyModifyEnd()
            
            # Save to temp file
            import tempfile
            import shutil
            
            with tempfile.NamedTemporaryFile(suffix=".fbx", delete=False) as tmp:
                temp_path = tmp.name
            
            exporter = fbx.FbxExporter.Create(manager, "")
            file_format = manager.GetIOPluginRegistry().GetNativeWriterFormat()
            if exporter.Initialize(temp_path, file_format, ios):
                exporter.Export(scene)
            exporter.Destroy()
            manager.Destroy()
            
            # Read the file and return it
            with open(temp_path, "rb") as f:
                fbx_data = f.read()
            os.remove(temp_path)
            
            filename = os.path.basename(full_path).replace(".fbx", "_inplace.fbx")
            return web.Response(
                body=fbx_data,
                content_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
            
        except ImportError:
            return web.json_response({"error": "FBX SDK not available"}, status=500)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

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
                "translate_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "translate_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "translate_z": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "rotate_x": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rotate_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rotate_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.01}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.01}),
                "scale_z": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_url",)
    FUNCTION = "load_model"
    CATEGORY = "HY-Motion/view"
    OUTPUT_NODE = True

    def load_model(self, model_path, translate_x=0.0, translate_y=0.0, translate_z=0.0, rotate_x=0.0, rotate_y=0.0, rotate_z=0.0, scale_x=1.0, scale_y=1.0, scale_z=1.0):
        def safe_float(v, default):
            try:
                if v is None or v == "": return default
                return float(v)
            except: return default

        tx = safe_float(translate_x, 0.0)
        ty = safe_float(translate_y, 0.0)
        tz = safe_float(translate_z, 0.0)
        rx = safe_float(rotate_x, 0.0)
        ry = safe_float(rotate_y, 0.0)
        rz = safe_float(rotate_z, 0.0)
        sx = safe_float(scale_x, 1.0)
        sy = safe_float(scale_y, 1.0)
        sz = safe_float(scale_z, 1.0)

        if model_path == "none":
            return (None,)
        
        ext = os.path.splitext(model_path)[1].lower()[1:]
        return {
            "ui": {
                "model_url": model_path, 
                "format": ext,
                "transform": {
                    "translate": [tx, ty, tz],
                    "rotate": [rx, ry, rz],
                    "scale": [sx, sy, sz]
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
