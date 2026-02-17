import os
import torch
import numpy as np
import smplx
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import folder_paths
import threading

from .hymotion.utils.geometry import rotation_matrix_to_rot6d
from .hymotion.utils.data_types import HYMotionData

# Configuration
LEARNING_RATES = (0.1, 0.01)
ITERATIONS = ((40, 80, 40), 300)

# =============================================================================
# MHR Formatting Utilities
# =============================================================================

def format_mhr_params(output: dict) -> dict:
    """Format single-frame output into MHR_PARAMS sequence format."""
    if output is None:
        return None
        
    # If this is a processed mesh_data/skeleton dict, it might have 'raw_output'
    if "raw_output" in output and isinstance(output["raw_output"], dict):
        output = output["raw_output"]

    # Map possible key names
    kp3d = output.get("pred_keypoints_3d")
    if kp3d is None:
        kp3d = output.get("joints") # mesh_data key
    if kp3d is None:
        kp3d = output.get("pred_joint_coords")
    if kp3d is None:
        kp3d = output.get("joint_positions") # skeleton key
        
    if kp3d is None:
        # If we still don't have keypoints, we can't do much, 
        # but let's try to at least get vertices so the next node doesn't crash on None
        vertices = output.get("pred_vertices")
        if vertices is None:
            vertices = output.get("vertices")
            
        if vertices is None:
            return None
        
        # Create dummy keypoints if we have vertices
        kp3d = np.zeros((70, 3))
            
    # MHR_PARAMS expects temporal sequences [F, ...]
    
    # Extract values with fallbacks
    kp2d = output.get("pred_keypoints_2d")
    if kp2d is None:
        kp2d = np.zeros((kp3d.shape[0], 2))
        
    vertices = output.get("pred_vertices")
    if vertices is None:
        vertices = output.get("vertices")
    if vertices is None:
        vertices = np.zeros((1, 3))
        
    cam_t = output.get("pred_cam_t")
    if cam_t is None:
        cam_t = output.get("camera")
    if cam_t is None:
        cam_t = np.zeros(3)
        
    global_rots = output.get("pred_global_rots")
    if global_rots is None:
        global_rots = output.get("joint_rotations")
    if global_rots is None:
        global_rots = np.eye(3)
        
    joint_coords = output.get("pred_joint_coords")
    if joint_coords is None:
        joint_coords = output.get("joint_positions")
    if joint_coords is None:
        joint_coords = kp3d
        
    pose_params = output.get("pose_params", {})
    
    body_pose = output.get("body_pose_params")
    if body_pose is None:
        body_pose = pose_params.get("body_pose")
    if body_pose is None:
        body_pose = np.zeros(260)
        
    hand_pose = output.get("hand_pose_params")
    if hand_pose is None:
        hand_pose = pose_params.get("hand_pose")
    if hand_pose is None:
        hand_pose = np.zeros(108)
        
    shape_params = output.get("shape_params")
    if shape_params is None:
        shape_params = pose_params.get("shape")
    if shape_params is None:
        shape_params = np.zeros(45)
        
    scale_params = output.get("scale_params")
    if scale_params is None:
        scale_params = pose_params.get("scale")
    if scale_params is None:
        scale_params = np.zeros(28)
        
    focal_length = output.get("focal_length")

    return {
        "type": "mhr",
        "num_frames": 1,
        "keypoints_3d": torch.from_numpy(kp3d).float().unsqueeze(0),
        "keypoints_2d": torch.from_numpy(kp2d).float().unsqueeze(0),
        "vertices": torch.from_numpy(vertices).float().unsqueeze(0),
        "cam_t": torch.from_numpy(cam_t).float().unsqueeze(0),
        "global_rots": torch.from_numpy(global_rots).float().unsqueeze(0),
        "joint_coords": torch.from_numpy(joint_coords).float().unsqueeze(0),
        "body_pose": torch.from_numpy(body_pose).float().unsqueeze(0),
        "hand_pose": torch.from_numpy(hand_pose).float().unsqueeze(0),
        "shape_params": torch.from_numpy(shape_params).float().unsqueeze(0),
        "scale_params": torch.from_numpy(scale_params).float().unsqueeze(0),
        "focal_length": focal_length,
    }

def stack_mhr_params(params_list: list) -> dict:
    """Stack a list of single-frame MHR_PARAMS into a single temporal sequence."""
    if not params_list:
        return None
        
    num_frames = len(params_list)
    
    # Initialize dictionary
    stacked = {
        "type": "mhr",
        "num_frames": num_frames,
    }
    
    # Keys to stack (all except type, num_frames, and focal_length)
    keys_to_stack = [
        "keypoints_3d", "keypoints_2d", "vertices", "cam_t", 
        "global_rots", "joint_coords", "body_pose", "hand_pose", 
        "shape_params", "scale_params"
    ]
    
    for key in keys_to_stack:
        tensors = [p[key] for p in params_list]
        stacked[key] = torch.cat(tensors, dim=0) # [F, ...]
        
    # Use focal length from the first frame
    stacked["focal_length"] = params_list[0].get("focal_length")
    
    return stacked

class HYMotionMHRLoader:
    """Load MHR vertices and camera translation from an NPZ file"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mhr_name": (folder_paths.get_filename_list("hymotion_npz"), {
                    "tooltip": "Select an MHR NPZ file from the output/hymotion_npz or input/hymotion_npz directory."
                }),
            }
        }
    
    RETURN_TYPES = ("MHR_PARAMS",)
    RETURN_NAMES = ("mhr_params",)
    FUNCTION = "load"
    CATEGORY = "HY-Motion/loaders"

    def load(self, mhr_name):
        full_path = folder_paths.get_full_path("hymotion_npz", mhr_name)
        if not full_path or not os.path.exists(full_path):
            raise FileNotFoundError(f"MHR file not found: {mhr_name}")
            
        print(f"[HY-Motion] Loading MHR Data: {full_path}")
        data = np.load(full_path, allow_pickle=True)
        
        if 'vertices' not in data or 'cam_t' not in data:
            raise ValueError(f"Selected file {mhr_name} is not a valid MHR export (missing 'vertices' or 'cam_t')")
            
        return ({"vertices": data['vertices'], "cam_t": data['cam_t']},)

class HYMotionMHRConverter:
    """
    Ultra-fast GPU-accelerated MHR to HyMotion converter.
    Fits SMPL-X parameters to MHR vertices in seconds.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mhr_params": ("MHR_PARAMS",),
                "fit_hands": ("BOOLEAN", {"default": True}),
                "flip_orientation": ("BOOLEAN", {"default": True}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("HYMOTION_DATA",)
    FUNCTION = "convert"
    CATEGORY = "HY-Motion/converters"

    def convert(self, mhr_params, fit_hands, flip_orientation, device):
        device_obj = torch.device(device)
        
        # 0. Set up paths relative to extension
        ext_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_file = os.path.join(ext_dir, "assets/mhr/mhr2smplx_mapping.npz")
        faces_file = os.path.join(ext_dir, "assets/mhr/mhr_faces_lod1.npy")
        smpl_model_dir = os.path.join(ext_dir, "assets/mhr/body_models")

        # 1. Prepare MHR vertices
        print(f"[HY-Motion] Processing MHR data...")
        vertices = mhr_params['vertices']
        cam_t = mhr_params['cam_t']
        
        # Handle both numpy (from loader) and torch (from live node)
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices)
        if isinstance(cam_t, np.ndarray):
            cam_t = torch.from_numpy(cam_t)
            
        vertices = vertices.float().to(device_obj)
        cam_t = cam_t.float().to(device_obj)
        
        flip_vec = torch.tensor([1, -1, -1] if flip_orientation else [1, 1, 1], device=device_obj).float()
        source_vertices = (vertices + cam_t[:, None, :]) * flip_vec
        num_frames = source_vertices.shape[0]

        # 2. Load Mapping and Surface info
        mapping = np.load(mapping_file)
        mapped_face_id = torch.from_numpy(mapping['triangle_ids']).long().to(device)
        baryc_coords = torch.from_numpy(mapping['baryc_coords']).float().to(device)
        source_faces = torch.from_numpy(np.load(faces_file).astype(np.int64)).to(device)

        # 3. Interpolate MHR vertices to SMPL topology
        triangles = source_vertices[:, source_faces[mapped_face_id], :]
        baryc_expanded = baryc_coords[None, :, :, None]
        target_vertices = (triangles * baryc_expanded).sum(dim=2)

        # Result container for thread
        output_results = [None]
        error_results = [None]

        def optimization_worker():
            try:
                # Inside a new thread, inference_mode is disabled by default
                with torch.enable_grad():
                    # 4. Initialize SMPLX Model
                    smpl_model = smplx.create(
                        model_path=smpl_model_dir,
                        model_type='smplx',
                        gender='neutral',
                        use_pca=False,
                        flat_hand_mean=True,
                        batch_size=num_frames
                    ).to(device_obj)

                    smpl_faces = torch.from_numpy(smpl_model.faces.astype(np.int64)).to(device_obj)
                    smpl_edges = torch.cat([smpl_faces[:, [0, 1]], smpl_faces[:, [1, 2]], smpl_faces[:, [2, 0]]], dim=0)
                    smpl_edges = torch.unique(torch.sort(smpl_edges, dim=1)[0], dim=0)

                    # 5. Define Trainable Variables
                    betas = torch.nn.Parameter(torch.zeros(1, 10, device=device_obj))
                    global_orient = torch.nn.Parameter(torch.zeros(num_frames, 3, device=device_obj))
                    body_pose = torch.nn.Parameter(torch.zeros(num_frames, 63, device=device_obj))
                    left_hand_pose = torch.nn.Parameter(torch.zeros(num_frames, 45, device=device_obj))
                    right_hand_pose = torch.nn.Parameter(torch.zeros(num_frames, 45, device=device_obj))
                    transl = torch.nn.Parameter(torch.zeros(num_frames, 3, device=device_obj))

                    zero_jaw = torch.zeros(num_frames, 3, device=device_obj)
                    zero_eyes = torch.zeros(num_frames, 6, device=device_obj)

                    smpl_model.train()

                    # 6. Stage 1: Initial Pose Optimization
                    print("[HY-Motion] Stage 1: Initial Pose Optimization...")
                    optimizable_configs = [["global_orient"], ["global_orient", "body_pose", "betas"], ["global_orient", "body_pose", "betas"]]
                    target_edge_vecs = target_vertices[:, smpl_edges[:, 1], :] - target_vertices[:, smpl_edges[:, 0], :]

                    for op_keys, iters in zip(optimizable_configs, ITERATIONS[0]):
                        vars_to_opt = []
                        if "global_orient" in op_keys: vars_to_opt.append(global_orient)
                        if "body_pose" in op_keys: vars_to_opt.append(body_pose)
                        if "betas" in op_keys: vars_to_opt.append(betas)
                        
                        optimizer = torch.optim.Adam(vars_to_opt, lr=LEARNING_RATES[0])
                        for _ in range(iters):
                            optimizer.zero_grad()
                            output = smpl_model(
                                betas=betas.expand(num_frames, -1),
                                global_orient=global_orient,
                                body_pose=body_pose,
                                transl=transl,
                                jaw_pose=zero_jaw,
                                leye_pose=zero_eyes[:, :3],
                                reye_pose=zero_eyes[:, 3:],
                                return_verts=True
                            )
                            est_edges = output.vertices[:, smpl_edges[:, 1], :] - output.vertices[:, smpl_edges[:, 0], :]
                            loss = torch.abs(est_edges - target_edge_vecs).mean()
                            loss.backward()
                            optimizer.step()

                    # Initial translation alignment
                    with torch.no_grad():
                        output = smpl_model(betas=betas.expand(num_frames, -1), global_orient=global_orient, body_pose=body_pose, transl=transl)
                        transl.data += (target_vertices - output.vertices).mean(dim=1)

                    # 7. Stage 2: Fine Optimization
                    print(f"[HY-Motion] Stage 2: Fine Optimization (Hands: {fit_hands})...")
                    params_to_opt = [betas, global_orient, body_pose, transl]
                    if fit_hands:
                        params_to_opt.extend([left_hand_pose, right_hand_pose])
                        
                    optimizer = torch.optim.Adam(params_to_opt, lr=LEARNING_RATES[1])
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
                    
                    for epoch in tqdm(range(ITERATIONS[1])):
                        optimizer.zero_grad()
                        output = smpl_model(
                            betas=betas.expand(num_frames, -1),
                            global_orient=global_orient,
                            body_pose=body_pose,
                            transl=transl,
                            left_hand_pose=left_hand_pose if fit_hands else None,
                            right_hand_pose=right_hand_pose if fit_hands else None,
                            jaw_pose=zero_jaw,
                            leye_pose=zero_eyes[:, :3],
                            reye_pose=zero_eyes[:, 3:],
                            return_verts=True
                        )
                        
                        edge_loss = torch.abs((output.vertices[:, smpl_edges[:, 1], :] - output.vertices[:, smpl_edges[:, 0], :]) - target_edge_vecs).mean()
                        vertex_loss = torch.square(output.vertices - target_vertices).mean()
                        
                        loss = (1.0 if epoch < 50 else 0) * edge_loss + vertex_loss
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    # 8. Final Formatting
                    print("[HY-Motion] Finalizing motion data...")
                    with torch.no_grad():
                        b_pose = body_pose.detach()
                        g_orient = global_orient.detach()
                        l_hand = left_hand_pose.detach()
                        r_hand = right_hand_pose.detach()
                        tr = transl.detach()
                        final_betas = betas.detach()

                        full_pose_aa = torch.zeros((num_frames, 52, 3), device=device_obj)
                        full_pose_aa[:, 0] = g_orient
                        full_pose_aa[:, 1:22] = b_pose.view(num_frames, 21, 3)
                        if fit_hands:
                            full_pose_aa[:, 22:37] = l_hand.view(num_frames, 15, 3)
                            full_pose_aa[:, 37:52] = r_hand.view(num_frames, 15, 3)
                        
                        full_pose_mat = torch.from_numpy(R.from_rotvec(full_pose_aa.cpu().view(-1, 3).numpy()).as_matrix()).view(num_frames, 52, 3, 3).to(device_obj)
                        rot6d = rotation_matrix_to_rot6d(full_pose_mat)
                        
                        final_output = smpl_model(
                            betas=final_betas.expand(num_frames, -1),
                            global_orient=g_orient,
                            body_pose=b_pose,
                            transl=tr,
                            left_hand_pose=l_hand if fit_hands else None,
                            right_hand_pose=r_hand if fit_hands else None,
                            return_verts=True
                        )
                        keypoints3d = final_output.joints[:, :52, :].detach()
                        
                        offset = tr[0:1].clone()
                        transl_relative = tr - offset
                        keypoints3d_relative = keypoints3d - offset[:, None, :]

                    output_results[0] = {
                        'rot6d': rot6d.cpu().unsqueeze(0), # [1, num_frames, 52, 6]
                        'transl': transl_relative.cpu().unsqueeze(0), # [1, num_frames, 3]
                        'keypoints3d': keypoints3d_relative.cpu().unsqueeze(0), # [1, num_frames, 52, 3]
                        'root_rotations_mat': full_pose_mat[:, 0].cpu().unsqueeze(0), # [1, num_frames, 3, 3]
                        'betas': torch.zeros((1, 16)),
                        'num_frames': num_frames,
                        'poses': torch.cat([g_orient, b_pose, l_hand, r_hand], dim=1).cpu().unsqueeze(0),
                        'Rh': g_orient.cpu().unsqueeze(0),
                        'trans': transl_relative.cpu().unsqueeze(0)
                    }
            except Exception as e:
                error_results[0] = e

        # Execute in separate thread to escape ComfyUI's inference_mode
        thread = threading.Thread(target=optimization_worker)
        thread.start()
        thread.join()

        if error_results[0]:
            raise error_results[0]

        motion_data = HYMotionData(
            output_dict=output_results[0],
            text=f"MHR GPU Converter",
            duration=num_frames / 30.0,
            seeds=[0],
            device_info=str(device_obj)
        )

        print("✓ [HY-Motion] MHR Conversion Complete!")
        return (motion_data,)

class SAM3DToHyMotion:
    """
    Unified node: Converts SAM 3D Body outputs directly to HyMotion data.
    Internally handles MHR formatting and SMPL-X fitting.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fit_hands": ("BOOLEAN", {"default": True}),
                "flip_orientation": ("BOOLEAN", {"default": True}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM 3D Body Process node"
                }),
                "skeleton": ("SKELETON", {
                    "tooltip": "Skeleton data from SAM 3D Body Process node"
                }),
            }
                
        }

    RETURN_TYPES = ("HYMOTION_DATA",)
    FUNCTION = "process"
    CATEGORY = "HY-Motion/converters"

    def process(self, fit_hands, flip_orientation, device, mesh_data=None, skeleton=None, mhr_params=None):
        # 1. Prepare MHR Params
        if mhr_params is None:
            if mesh_data is None and skeleton is None:
                raise ValueError("Must provide either mesh_data, skeleton, or mhr_params")
            
            source = mesh_data if mesh_data is not None else skeleton
            if isinstance(source, list):
                # Batch conversion
                params_list = []
                for item in source:
                    p = format_mhr_params(item)
                    if p:
                        params_list.append(p)
                if not params_list:
                    return (None,)
                mhr_params = stack_mhr_params(params_list)
            else:
                mhr_params = format_mhr_params(source)

        if mhr_params is None:
            return (None,)

        # 2. Convert using HYMotionMHRConverter logic
        converter = HYMotionMHRConverter()
        return converter.convert(mhr_params, fit_hands, flip_orientation, device)

NODE_CLASS_MAPPINGS_MHR = {
    "HYMotionMHRLoader": HYMotionMHRLoader,
    "HYMotionMHRConverter": HYMotionMHRConverter,
    "SAM3DToHyMotion": SAM3DToHyMotion,
}

NODE_DISPLAY_NAME_MAPPINGS_MHR = {
    "HYMotionMHRLoader": "HY-Motion MHR Loader",
    "HYMotionMHRConverter": "HY-Motion MHR Converter",
    "SAM3DToHyMotion": "SAM 3D Body: SAM3D to HyMotion",
}
