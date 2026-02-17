
import os
import sys
import numpy as np

# Add the project root to sys.path
sys.path.append(os.getcwd())

from hymotion.utils.retarget_fbx import load_fbx, get_skeleton_height, extract_animation, load_bone_mapping, retarget_animation

def verify_rig_agnostic(src_path, tgt_path):
    print(f"Source: {src_path}")
    print(f"Target Template: {tgt_path}")
    
    # 1. Load Source FBX (The motion provider)
    _, src_scene, src_skel = load_fbx(src_path)
    extract_animation(src_scene, src_skel)
    
    # 2. Load Target FBX (The skeleton provider)
    _, tgt_scene, tgt_skel = load_fbx(tgt_path)
    
    # 3. Perform Mapping
    mapping = load_bone_mapping("", src_skel, tgt_skel)
    
    # 4. Run Retargeting with NEW logic
    print("\nRunning Retargeting with NEW Rig-Agnostic Logic...")
    rots, locs, active = retarget_animation(
        src_skel, tgt_skel, mapping, 
        force_scale=1.0, 
        auto_stride=True
    )
    
    # Check for movement in 'bone_0' (identified as root)
    target_root_name = None
    for bname in ['bone_0', 'hips', 'pelvis']:
        if tgt_skel.get_bone_case_insensitive(bname):
            target_root_name = tgt_skel.get_bone_case_insensitive(bname).name
            break
            
    if not target_root_name or target_root_name not in locs:
        print(f"Error: Could not find root ({target_root_name}) in retargeted locs.")
        return

    frames = sorted(locs[target_root_name].keys())
    start_f = frames[0]
    end_f = frames[-1]
    
    start_y = locs[target_root_name][start_f][1]
    end_y = locs[target_root_name][end_f][1]
    
    print(f"\nTrajectory Analysis for {target_root_name} (Y-axis):")
    print(f"{start_f:>6} | {start_y:>12.4f}")
    print(f"{end_f:>6} | {end_y:>12.4f}")

    movement = np.abs(end_y - start_y)
    print(f"\nVertical Movement Delta: {end_y - start_y:.4f}")
    if movement > 0.001:
        print("SUCCESS: Root bone is correctly animated!")
    else:
        print("FAILURE: Character is still static.")

if __name__ == "__main__":
    src = "/home/aero/comfy/ComfyUI/output/hymotion_fbx/originalone_20260124_092207782_991ef242_000.fbx"
    # This is the original character rig (Human_Croc which uses UniRig bones)
    tgt = "/home/aero/comfy/ComfyUI/input/Human_Croc.fbx"
    verify_rig_agnostic(src, tgt)
