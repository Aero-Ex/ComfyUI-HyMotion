import sys
import os

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from hymotion.utils.retarget_fbx import load_fbx, get_skeleton_height

def main():
    target_file = "/home/aero/comfy/ComfyUI/output/hymotion_fbx/finaltesting_20260123_192617672_521133e6_000.fbx"
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found")
        return
        
    print(f"Inspecting Height: {target_file}")
    man, scene, skel = load_fbx(target_file)
    # We need the height of the BIND pose (rest pose)
    h = get_skeleton_height(skel)
    unit = scene.GetGlobalSettings().GetSystemUnit().GetScaleFactor()
    
    print(f"  Calculated Height: {h:.4f} units")
    print(f"  Unit Scale: {unit}")
    print(f"  Height in CM: {h * unit:.4f}")

if __name__ == "__main__":
    main()
