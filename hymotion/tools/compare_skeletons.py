import os
import sys

# Add the hymotion directory to path to import retarget_fbx
sys.path.append("/home/aero/comfy/ComfyUI/custom_nodes/ComfyUI-HyMotion")

try:
    from hymotion.utils.retarget_fbx import HAS_FBX_SDK
    if not HAS_FBX_SDK:
        print("FBX SDK not found. Please ensure it is installed.")
        sys.exit(1)
    
    import fbx
    from fbx import FbxManager, FbxScene, FbxImporter, FbxNodeAttribute
except ImportError as e:
    print(f"Error importing FBX SDK or utilities: {e}")
    sys.exit(1)

def get_skeleton_hierarchy(fbx_path):
    manager = FbxManager.Create()
    importer = FbxImporter.Create(manager, "")
    
    if not importer.Initialize(fbx_path, -1, manager.GetIOSettings()):
        print(f"Failed to initialize importer for {fbx_path}")
        return None
        
    scene = FbxScene.Create(manager, "Scene")
    importer.Import(scene)
    importer.Destroy()
    
    root_node = scene.GetRootNode()
    hierarchy = []
    
    def walk_nodes(node, depth=0):
        if not node:
            return
            
        attr = node.GetNodeAttribute()
        is_bone = False
        if attr:
            at = attr.GetAttributeType()
            # 3 = eSkeleton, 4 = eLimbNode, 2 = eNull
            if at in [3, 4]: 
                is_bone = True
            elif at == 2 and (node.GetChildCount() > 0):
                is_bone = True
        
        # Keyword-based fallback consistent with retarget_fbx.py
        name_lower = node.GetName().lower()
        keywords = ['hips', 'hip', 'spine', 'neck', 'head', 'arm', 'leg', 'foot', 'ankle', 'knee', 'shoulder', 'elbow', 'pelvis', 'joint', 'mixamo', 'thigh', 'upper', 'forearm', 'hand', 'finger', 'clavicle', 'collar', 'toe', 'thumb', 'index', 'middle', 'ring', 'pinky', 'upleg', 'downleg', 'wrist', 'chest', 'belly', 'bone_', 'character', 'calf', 'ball', 'shin', 'mid', 'metacarpal', 'subclavian']
        if any(k in name_lower for k in keywords):
            is_bone = True
            
        if is_bone:
            parent = node.GetParent()
            parent_name = parent.GetName() if parent else "None"
            hierarchy.append({
                "name": node.GetName(),
                "parent": parent_name,
                "depth": depth
            })
            
        for i in range(node.GetChildCount()):
            walk_nodes(node.GetChild(i), depth + (1 if is_bone else 0))
            
    walk_nodes(root_node)
    manager.Destroy()
    return hierarchy

def dump_hierarchy(fbx_path, output_path):
    print(f"Processing {fbx_path}...")
    hierarchy = get_skeleton_hierarchy(fbx_path)
    if hierarchy:
        with open(output_path, "w") as f:
            for bone in hierarchy:
                indent = "  " * bone["depth"]
                f.write(f"{indent}{bone['name']} (Parent: {bone['parent']})\n")
        print(f"Hierarchy dumped to {output_path}")

if __name__ == "__main__":
    fbx1 = "/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx"
    fbx2 = "/home/aero/Documents/humang9.fbx"
    
    dump_hierarchy(fbx1, "g3f_hierarchy.txt")
    dump_hierarchy(fbx2, "g9_hierarchy.txt")
