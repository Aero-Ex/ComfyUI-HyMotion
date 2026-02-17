import sys
import os
import numpy as np

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from hymotion.utils.retarget_fbx import load_fbx, extract_animation

def get_bone_movement(filepath):
    print(f"Loading: {os.path.basename(filepath)}")
    man, scene, skel = load_fbx(filepath)
    extract_animation(scene, skel)
    
    hips = skel.get_bone_case_insensitive('mixamorig:Hips')
    if not hips: return None
        
    frames = sorted(hips.world_location_animation.keys())
    pos = [hips.world_location_animation[f] for f in frames]
    return np.array(pos)

def main():
    fbx_file = "/home/aero/comfy/ComfyUI/output/test_retarget_fix.fbx"
    npz_file = "/home/aero/comfy/ComfyUI/output/test_retarget_fixnpz.fbx"
    
    pos1 = get_bone_movement(fbx_file)
    pos2 = get_bone_movement(npz_file)
    
    if pos1 is None or pos2 is None: return
    
    print(f"\nFrame | {'FBX Hips (X, Z)':<20} | {'NPZ Hips (X, Z)':<20}")
    n = min(len(pos1), len(pos2), 100)
    for i in range(n):
        if i % 20 == 0 or i == n-1:
            p1 = [pos1[i,0], pos1[i,2]]
            p2 = [pos2[i,0], pos2[i,2]]
            print(f"{i:<5} | {str(p1):<20} | {str(p2):<20}")

    travel1 = np.linalg.norm(pos1[-1] - pos1[0])
    travel2 = np.linalg.norm(pos2[-1] - pos2[0])
    print(f"\nTotal Travel (Entire Range):")
    print(f"  FBX Output: {travel1:.2f} cm")
    print(f"  NPZ Output: {travel2:.2f} cm")
    print(f"  Final Velocity Ratio (NPZ/FBX): {travel2/travel1:.4f}" if travel1 > 0 else "  N/A")

if __name__ == "__main__":
    main()
