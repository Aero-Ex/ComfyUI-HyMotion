import os
import json
import torch
import numpy as np
import folder_paths
import time
from server import PromptServer
from aiohttp import web

# Shared data types
from .hymotion.utils.data_types import HYMotionData, HYMotionFrame

# Retargeting imports
try:
    from .hymotion.utils.retarget_fbx import (
        Skeleton, BoneData, load_npz, load_fbx, load_bone_mapping,
        retarget_animation, apply_retarget_animation, save_fbx,
        collect_skeleton_nodes, extract_animation, get_skeleton_height
    )
    HAS_RETARGET_UTILS = True
except ImportError:
    HAS_RETARGET_UTILS = False

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

@PromptServer.instance.routes.get("/hymotion/get_assets")
async def get_assets(request):
    """
    Returns a lists of available FBX files (input/output) and templates.
    """
    try:
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        
        fbx_files = []
        templates = []
        extensions = [".fbx", ".glb", ".gltf", ".obj"]
        
        def scan_dir(base_path, prefix="", target_list=None):
            if not os.path.exists(base_path): return
            if target_list is None: target_list = fbx_files
            for root, _, filenames in os.walk(base_path):
                for f in filenames:
                    if any(f.lower().endswith(ext) for ext in extensions):
                        rel = os.path.relpath(os.path.join(root, f), base_path).replace("\\", "/")
                        target_list.append(f"{prefix}{rel}")

        scan_dir(input_dir, "input/")
        scan_dir(output_dir, "output/")
        
        # Also scan templates
        from .__init__ import CURRENT_DIR
        template_dir = os.path.join(CURRENT_DIR, "assets", "wooden_models")
        scan_dir(template_dir, "", templates)

        return web.json_response({
            "fbx_files": sorted(fbx_files) if fbx_files else ["none"],
            "templates": sorted(templates) if templates else ["none"]
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

class HYMotion3DModelLoader:
    """
    Standalone viewer for 3D models (FBX, GLB, GLTF, OBJ).
    Focuses on model loading and manual posing.
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
                    "tooltip": "Select a 3D model to load and pose"
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
                # HIDDEN: Frame inputs in development
                # "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                # "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                # HIDDEN: Pose capture in development
                # "start_pose_json": ("STRING", {"default": "{}"}),
                # "end_pose_json": ("STRING", {"default": "{}"}),
            },
        }

    # HIDDEN: Pose outputs disabled for now
    # RETURN_TYPES = ("STRING", "HYMOTION_FRAME", "HYMOTION_FRAME", "HYMOTION_DATA")
    # RETURN_NAMES = ("model_path", "start_pose", "end_pose", "motion_data")
    RETURN_TYPES = ("STRING", "HYMOTION_DATA")
    RETURN_NAMES = ("model_path", "motion_data")
    FUNCTION = "load_model"
    CATEGORY = "HY-Motion/view"
    OUTPUT_NODE = True

    def load_model(self, model_path, translate_x=0.0, translate_y=0.0, translate_z=0.0, 
                   rotate_x=0.0, rotate_y=0.0, rotate_z=0.0, 
                   scale_x=1.0, scale_y=1.0, scale_z=1.0,
                   start_frame=0, end_frame=0,
                   start_pose_json="{}", end_pose_json="{}"):
        print(f"[HY-Motion] Loader: {model_path} | Frames: {start_frame} to {end_frame}")
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
            return (None, None, None)
        
        # Resolve full path
        full_path = ""
        if model_path.startswith("input/"):
            full_path = os.path.join(folder_paths.get_input_directory(), model_path[6:])
        elif model_path.startswith("output/"):
            full_path = os.path.join(folder_paths.get_output_directory(), model_path[7:])
        else:
            full_path = model_path

        # Handle Manual Rig Poses (Start/End)
        def parse_pose(json_str, name="Pose"):
            if not json_str or json_str == "{}":
                print(f"[HY-Motion] {name}: No pose data (empty)")
                return None
            try:
                import json
                from .hymotion.utils.geometry import rotation_matrix_to_rot6d, quaternion_to_matrix
                
                pose_data = json.loads(json_str)
                if not pose_data or "bones" not in pose_data or not pose_data["bones"]:
                    print(f"[HY-Motion] {name}: Invalid pose data (no bones)")
                    return None
                
                #SMPL-H 52 joints
                smpl_names = [
                    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3'
                ]
                
                rot6d_p = torch.zeros((52, 6))
                # Default to identity (1,0,0, 0,1,0)
                rot6d_p[:, 0] = 1.0
                rot6d_p[:, 4] = 1.0
                
                bone_data = pose_data["bones"]
                matched_count = 0
                for i, name_joint in enumerate(smpl_names):
                    # Fuzzy matching for bone names
                    # 1. Direct match
                    # 2. Lowercase match
                    # 3. Match without prefix (e.g. "mixamorig:Pelvis" -> "pelvis")
                    matched_key = None
                    # Clean up the target name (e.g. "left_hip" -> "lefthip")
                    name_clean = name_joint.lower().replace('_', '').replace(' ', '')
                    
                    for k in bone_data.keys():
                        # Clean up the key from the model (e.g. "mixamorig:L_Hip" -> "lhip")
                        k_clean = k.lower().split(':')[-1].replace('_', '').replace(' ', '')
                        # Handle common abbreviations and synonyms
                        if k_clean.startswith('l'): k_clean = 'left' + k_clean[1:]
                        if k_clean.startswith('r'): k_clean = 'right' + k_clean[1:]
                        
                        if k_clean == name_clean or (name_clean == "pelvis" and k_clean == "hips"):
                            matched_key = k
                            break
                    
                    if matched_key:
                        matched_count += 1
                        b = bone_data[matched_key]
                        # rot can be [9] (flat matrix) or [3, 3] or [4] (quat)
                        rot_raw = b["rot"]
                        if len(rot_raw) == 9:
                            mat = torch.tensor(rot_raw).float().view(3, 3)
                        elif len(rot_raw) == 4:
                            # Quat wxyz or xyzw? Three.js uses xyzw
                            q = torch.tensor([rot_raw[3], rot_raw[0], rot_raw[1], rot_raw[2]])
                            mat = quaternion_to_matrix(q.unsqueeze(0))[0]
                        else:
                            mat = torch.tensor(rot_raw).float()
                        
                        r6 = rotation_matrix_to_rot6d(mat.unsqueeze(0))[0]
                        rot6d_p[i] = r6
                
                # Root position (normalized to meters)
                root_pos = torch.tensor(pose_data.get("root_pos", [0, 0, 0])).float()
                
                print(f"[HY-Motion] {name}: Extracted {matched_count}/52 bones. Root Pos: {root_pos.tolist()}")
                
                return HYMotionFrame(rot6d=rot6d_p.unsqueeze(0), transl=root_pos.unsqueeze(0))
            except Exception as e:
                print(f"[HY-Motion] {name}: Error parsing pose JSON: {e}")
                import traceback
                traceback.print_exc()
                return None

        start_pose = parse_pose(start_pose_json, "Start Pose")
        end_pose = parse_pose(end_pose_json, "End Pose")

        ext = os.path.splitext(model_path)[1].lower()[1:]
        # Create a motion data object if both poses exist
        motion_data = None
        if start_pose is not None and end_pose is not None:
            # Combine into a 2-frame sequence
            rot6d = torch.cat([start_pose.rot6d, end_pose.rot6d], dim=0).unsqueeze(0) # [1, 2, 52, 6]
            transl = torch.cat([start_pose.transl, end_pose.transl], dim=0).unsqueeze(0) # [1, 2, 3]
            
            output_dict = {
                "rot6d": rot6d[:, :, :22], # Truncate to 22 joints for the Sampler
                "transl": transl,
            }
            
            # Use WoodenMesh to compute keypoints if possible
            try:
                from .nodes_modular import _WOODEN_MESH_CACHE, WoodenMesh, construct_smpl_data_dict
                global _WOODEN_MESH_CACHE
                if _WOODEN_MESH_CACHE is None:
                    _WOODEN_MESH_CACHE = WoodenMesh()
                
                # Use forward_batch to compute keypoints for the 2-frame sequence
                # We use the full 52 joints for high-fidelity keypoint visualization
                res = _WOODEN_MESH_CACHE.forward_batch({"rot6d": rot6d, "trans": transl})
                keypoints = res["keypoints3d"] # [1, 2, 52, 3]
                output_dict["keypoints3d"] = keypoints
            except Exception as e:
                print(f"[HY-Motion] Loader: Could not compute keypoints for motion_data: {e}")

            motion_data = HYMotionData(
                output_dict=output_dict,
                text=f"Keyframes: {os.path.basename(model_path)}",
                duration=2.0 / 30.0,
                seeds=[0],
                device_info="cpu"
            )

        return {
            "ui": {
                "model_url": [model_path], 
                "format": [ext],
                "start_frame": [start_frame],
                "end_frame": [end_frame],
                "transform": [{
                    "translate": [tx, ty, tz],
                    "rotate": [rx, ry, rz],
                    "scale": [sx, sy, sz]
                }],
                "timestamp": [time.time()]
            }, 
            # HIDDEN: Pose outputs removed
            # "result": (model_path, start_pose, end_pose, motion_data)
            "result": (model_path, motion_data)
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
        
        # Handle newline-separated paths (multi-batch)
        paths = selected.split('\n')
        prefixed_paths = []
        for p in paths:
            p = p.strip()
            if not p: continue
            if not p.startswith("output/") and not p.startswith("input/"):
                p = f"output/{p}"
            prefixed_paths.append(p.replace("\\", "/"))
        
        full_path = "\n".join(prefixed_paths)
        return {"ui": {"fbx_url": [full_path], "timestamp": [time.time()]}, "result": (full_path,)}

NODE_CLASS_MAPPINGS = {
    "HYMotionFBXPlayer": HYMotionFBXPlayer,
    "HYMotion3DModelLoader": HYMotion3DModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionFBXPlayer": "HY-Motion 3D FBX Player (Legacy)",
    "HYMotion3DModelLoader": "HY-Motion 3D Model Loader"
}
