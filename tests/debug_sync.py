import sys
import os
import numpy as np

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from hymotion.utils.retarget_fbx import load_fbx, load_npz

def inspect(filepath):
    print(f"\nInspecting: {os.path.basename(filepath)}")
    if filepath.endswith('.npz'):
        skel = load_npz(filepath)
    else:
        man, scene, skel = load_fbx(filepath)
    
    frames = list(range(skel.frame_start, skel.frame_end + 1))
    if not frames:
        print("  Empty animation!")
        return

    print(f"  Frame Range: {skel.frame_start} to {skel.frame_end} (Total: {len(frames)})")
    
    if not filepath.endswith('.npz'):
        mode = scene.GetGlobalSettings().GetTimeMode()
        print(f"  FBX Time Mode: {mode}")
        unit = scene.GetGlobalSettings().GetSystemUnit()
        print(f"  Unit Scale Factor: {unit.GetScaleFactor()}")
        print(f"  Unit System: {unit.GetSystemUnitDescription()}")
        
        stack = scene.GetCurrentAnimationStack()
        if stack:
            layer = stack.GetMember(0)
            span = stack.GetLocalTimeSpan()
            print(f"  Duration: {span.GetDuration().GetSecondCount()}s")
            
            # Find root node manually
            root_candidates = ["Hips", "Pelvis", "mixamorig:Hips"]
            root_node = None
            for rc in root_candidates:
                root_node = scene.FindNodeByName(rc)
                if root_node: break
                
            if root_node:
                curve_x = root_node.LclTranslation.GetCurve(layer, "X")
                curve_z = root_node.LclTranslation.GetCurve(layer, "Z")
                if curve_x and curve_z:
                    v_start = [curve_x.KeyGetValue(0), 0, curve_z.KeyGetValue(0)]
                    v_end = [curve_x.KeyGetValue(curve_x.KeyGetCount()-1), 0, curve_z.KeyGetValue(curve_z.KeyGetCount()-1)]
                    move = np.linalg.norm(np.array(v_end) - np.array(v_start))
                    print(f"  Root Raw Move: {move:.4f} units")
                    print(f"  Root Move in CM: {move * unit.GetScaleFactor():.4f}")
    
    print(f"  FPS: {skel.fps}")
    
    # Check root bone position at start
    root_names = ['pelvis', 'hips', 'bone_0', 'mixamorig:hips']
    root = None
    for n in root_names:
        root = skel.get_bone_case_insensitive(n)
        if root: break
        
    if root:
        pos_0 = root.world_location_animation.get(frames[0])
        print(f"  Root Pos at Frame {frames[0]}: {pos_0}")
    else:
        print("  Root bone not found!")

def main():
    files = [
        "/home/aero/comfy/ComfyUI/output/hymotion_npz/wakingfromdie_20260123_182552595_c2e073c9_000.npz",
        "/home/aero/comfy/ComfyUI/output/hymotion_fbx/finaltesting_20260123_192617672_521133e6_000.fbx"
    ]
    for f in files:
        if os.path.exists(f):
            inspect(f)
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    main()
