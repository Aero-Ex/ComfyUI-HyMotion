import os
import torch
import numpy as np
import smplx
from tqdm import tqdm
import trimesh

# Configuration (Exact copies from official tool)
LEARNING_RATES = (0.1, 0.01)
ITERATIONS = ((40, 80, 40), 300)
CENTIMETERS_TO_METERS = 0.01
METERS_TO_CENTIMETERS = 100.0

# Paths
MHR_CONV_DIR = "/home/aero/comfy/ComfyUI/MHR/tools/mhr_smpl_conversion"
MAPPING_FILE = os.path.join(MHR_CONV_DIR, "assets/mhr2smplx_mapping.npz")
INPUT_VERTS = "/home/aero/comfy/ComfyUI/output/mhr_motion.npz"
OUTPUT_NPZ = "/home/aero/comfy/ComfyUI/output/smpl_conversion/smpl_motion_gpu.npz"
# Point to parent so it finds 'smplx/SMPLX_NEUTRAL.npz'
SMPLX_MODEL_DIR = "/home/aero/comfy/ComfyUI/models/motion_capture/body_models"

def gpu_fit():
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # 1. Load MHR vertices (cm world space, matching official tool)
    print(f"Loading MHR vertices from {INPUT_VERTS}...")
    mhr_data = np.load(INPUT_VERTS, allow_pickle=True)
    vertices = torch.from_numpy(mhr_data['vertices']).float().to(device)
    cam_t = torch.from_numpy(mhr_data['cam_t']).float().to(device)
    
    # Process all frames and flip Y/Z (OpenCV -> OpenGL/SMPL space)
    source_vertices = (vertices + cam_t[:, None, :]) * torch.tensor([1, -1, -1], device=device)
    num_frames = source_vertices.shape[0]
    print(f"✓ Loaded {num_frames} frames")

    # 2. Load Mapping and Surface info
    print("Loading vertex mapping and topology...")
    mapping = np.load(MAPPING_FILE)
    mapped_face_id = torch.from_numpy(mapping['triangle_ids']).long().to(device)
    baryc_coords = torch.from_numpy(mapping['baryc_coords']).float().to(device)
    
    # Needs source faces (MHR faces) for interpolation
    # Extracted previously to avoid trimesh FBX limitations
    mhr_faces_path = "/home/aero/comfy/ComfyUI/custom_nodes/ComfyUI-HyMotion/mhr_faces_lod1.npy"
    if not os.path.exists(mhr_faces_path):
        raise FileNotFoundError(f"MHR faces not found at {mhr_faces_path}. Please run the extraction command first.")
    source_faces = torch.from_numpy(np.load(mhr_faces_path).astype(np.int64)).to(device)

    # 3. Interpolate MHR vertices to SMPL topology
    print("Mapping MHR vertices to SMPL topology...")
    # triangles: [B, N_target, 3, 3]
    triangles = source_vertices[:, source_faces[mapped_face_id], :]
    # baryc: [1, N_target, 3, 1]
    baryc_expanded = baryc_coords[None, :, :, None]
    target_vertices = (triangles * baryc_expanded).sum(dim=2)
    print(f"✓ Target vertices mapped: {target_vertices.shape}")

    # 4. Initialize SMPLX Model
    print("Initializing SMPLX model...")
    smpl_model = smplx.create(
        model_path=SMPLX_MODEL_DIR,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True,
        batch_size=num_frames
    ).to(device)

    smpl_faces = torch.from_numpy(smpl_model.faces.astype(np.int64)).to(device)
    smpl_edges = torch.cat([smpl_faces[:, [0, 1]], smpl_faces[:, [1, 2]], smpl_faces[:, [2, 0]]], dim=0)
    smpl_edges = torch.unique(torch.sort(smpl_edges, dim=1)[0], dim=0)

    # 5. Define Trainable Variables
    # SMPLX body pose (21 joints) + hands (15 joints * 2) + global + betas
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    global_orient = torch.zeros(num_frames, 3, device=device, requires_grad=True)
    body_pose = torch.zeros(num_frames, 63, device=device, requires_grad=True)
    left_hand_pose = torch.zeros(num_frames, 45, device=device, requires_grad=True)
    right_hand_pose = torch.zeros(num_frames, 45, device=device, requires_grad=True)
    transl = torch.zeros(num_frames, 3, device=device, requires_grad=True)

    # Dummy parameters for SMPLX (required for forward)
    zero_jaw = torch.zeros(num_frames, 3, device=device)
    zero_eyes = torch.zeros(num_frames, 6, device=device)

    # 6. Stage 1: Initial Pose Optimization
    print("Stage 1: Initial Pose Optimization...")
    optimizable_configs = [
        ["global_orient"],
        ["global_orient", "body_pose", "betas"],
        ["global_orient", "body_pose", "betas"], 
    ]
    
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
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                leye_pose=zero_eyes[:, :3],
                reye_pose=zero_eyes[:, 3:],
                return_verts=True
            )
            est_edges = output.vertices[:, smpl_edges[:, 1], :] - output.vertices[:, smpl_edges[:, 0], :]
            loss = torch.abs(est_edges - target_edge_vecs).mean()
            loss.backward()
            optimizer.step()

    # Initial translation
    with torch.no_grad():
        output = smpl_model(betas=betas.expand(num_frames, -1), global_orient=global_orient, body_pose=body_pose, transl=transl)
        transl += (target_vertices - output.vertices).mean(dim=1)

    # 7. Stage 2: Fine Optimization
    print("Stage 2: Fine Optimization (Including Hands)...")
    optimizer = torch.optim.Adam([betas, global_orient, body_pose, transl, left_hand_pose, right_hand_pose], lr=LEARNING_RATES[1])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    
    for epoch in tqdm(range(ITERATIONS[1])):
        optimizer.zero_grad()
        output = smpl_model(
            betas=betas.expand(num_frames, -1),
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            jaw_pose=zero_jaw,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            leye_pose=zero_eyes[:, :3],
            reye_pose=zero_eyes[:, 3:],
            return_verts=True
        )
        
        # Edge Loss
        est_edges = output.vertices[:, smpl_edges[:, 1], :] - output.vertices[:, smpl_edges[:, 0], :]
        edge_loss = torch.abs(est_edges - target_edge_vecs).mean()
        
        # Vertex Loss
        vertex_loss = torch.square(output.vertices - target_vertices).mean()
        
        edge_weight = 1.0 if epoch < 50 else 0
        loss = edge_weight * edge_loss + vertex_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 8. Save
    print(f"Saving GPU results (with hand pose) to {OUTPUT_NPZ}...")
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ,
        betas=betas.detach().cpu().numpy().repeat(num_frames, axis=0),
        global_orient=global_orient.detach().cpu().numpy(),
        body_pose=body_pose.detach().cpu().numpy(),
        left_hand_pose=left_hand_pose.detach().cpu().numpy(),
        right_hand_pose=right_hand_pose.detach().cpu().numpy(),
        transl=transl.detach().cpu().numpy()
    )
    print("✓ GPU fitting complete!")

if __name__ == "__main__":
    gpu_fit()
