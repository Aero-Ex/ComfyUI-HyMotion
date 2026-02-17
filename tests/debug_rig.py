import sys
import os

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from hymotion.utils.retarget_fbx import load_fbx

def main():
    target_file = "/home/aero/comfy/ComfyUI/output/rigatoni_mia.fbx"
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found")
        return
        
    print(f"Inspecting: {target_file}")
    man, scene, skel = load_fbx(target_file)
    
    # Check root bone
    root_names = ['pelvis', 'hips', 'root', 'cog']
    root_bone = None
    for rn in root_names:
        root_bone = skel.get_bone_case_insensitive(rn)
        if root_bone: break
        
    if root_bone:
        print(f"Root Bone found: {root_bone.name}")
        print(f"  Rest Position: {root_bone.head}")
        print(f"  Parent: {root_bone.parent_name}")
        
        if root_bone.parent_name:
            p_mat = skel.node_world_matrices.get(root_bone.parent_name)
            if p_mat is not None:
                print(f"  Parent World Position: {p_mat[3, :3]}")
    else:
        print("Root bone not found!")
        
    # Check feet for floor level
    lfoot = skel.get_bone_case_insensitive('leftfoot') or skel.get_bone_case_insensitive('l_foot')
    if lfoot:
        print(f"Left Foot found: {lfoot.name}")
        print(f"  Rest Position: {lfoot.head}")

    print("\nSkeleton Outline (first 5 nodes):")
    for i, (name, node) in enumerate(list(skel.all_nodes.items())[:5]):
        wmat = skel.node_world_matrices.get(name)
        pos = wmat[3, :3] if wmat is not None else "N/A"
        print(f"  {name}: {pos}")

if __name__ == "__main__":
    main()
