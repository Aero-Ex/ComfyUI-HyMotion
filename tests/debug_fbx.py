"""
Debug script to inspect FBX bone hierarchy and positions.
Usage: python debug_fbx.py <path_to_fbx>
"""
import sys
import os

# Add FBX SDK to path if needed
try:
    import fbx
    from fbx import *
except ImportError:
    print("ERROR: FBX SDK not found. Please install fbx-sdk.")
    sys.exit(1)

def get_world_position(node):
    """Get the world position of a node."""
    try:
        mat = node.EvaluateGlobalTransform()
        return [mat.GetT()[0], mat.GetT()[1], mat.GetT()[2]]
    except:
        return [0, 0, 0]

def print_hierarchy(node, depth=0, parent_name=None):
    """Recursively print the bone hierarchy with positions."""
    name = node.GetName()
    pos = get_world_position(node)
    
    # Get node type
    attr = node.GetNodeAttribute()
    node_type = "Node"
    if attr:
        type_id = attr.GetAttributeType()
        type_map = {
            FbxNodeAttribute.eSkeleton: "Skeleton",
            FbxNodeAttribute.eMesh: "Mesh",
            FbxNodeAttribute.eNull: "Null",
            FbxNodeAttribute.eMarker: "Marker",
        }
        node_type = type_map.get(type_id, f"Type_{type_id}")
    
    # Count children
    child_count = node.GetChildCount()
    
    indent = "  " * depth
    print(f"{indent}[{node_type}] {name}")
    print(f"{indent}  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    print(f"{indent}  Parent: {parent_name or 'None'}")
    print(f"{indent}  Children: {child_count}")
    
    # List children
    if child_count > 0:
        children_names = [node.GetChild(i).GetName() for i in range(child_count)]
        print(f"{indent}  Child names: {children_names}")
    print()
    
    # Recurse
    for i in range(child_count):
        print_hierarchy(node.GetChild(i), depth + 1, name)

def analyze_fbx(filepath):
    """Analyze an FBX file and print detailed bone information."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print(f"{'='*60}\n")
    
    # Initialize FBX SDK
    manager = FbxManager.Create()
    ios = FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    
    # Load the FBX file
    importer = FbxImporter.Create(manager, "")
    if not importer.Initialize(filepath, -1, manager.GetIOSettings()):
        print(f"ERROR: Failed to load FBX: {importer.GetStatus().GetErrorString()}")
        return
    
    scene = FbxScene.Create(manager, "scene")
    importer.Import(scene)
    importer.Destroy()
    
    # Get basic info
    root = scene.GetRootNode()
    total_nodes = 0
    skeleton_nodes = []
    mesh_nodes = []
    
    def count_nodes(node):
        nonlocal total_nodes
        total_nodes += 1
        attr = node.GetNodeAttribute()
        if attr:
            if attr.GetAttributeType() == FbxNodeAttribute.eSkeleton:
                skeleton_nodes.append(node.GetName())
            elif attr.GetAttributeType() == FbxNodeAttribute.eMesh:
                mesh_nodes.append(node.GetName())
        for i in range(node.GetChildCount()):
            count_nodes(node.GetChild(i))
    
    count_nodes(root)
    
    print(f"Total Nodes: {total_nodes}")
    print(f"Skeleton Bones: {len(skeleton_nodes)}")
    print(f"Mesh Nodes: {len(mesh_nodes)}")
    print()
    
    # Check for UniRig detection criteria
    has_bone_0 = any("bone_0" in n.lower() for n in skeleton_nodes)
    has_armature = any("armature" in n.lower() for n in skeleton_nodes)
    print(f"UniRig Detection:")
    print(f"  Has 'bone_0': {has_bone_0}")
    print(f"  Has 'Armature': {has_armature}")
    print()
    
    # Print bone list
    print("Skeleton Bones:")
    for i, bone in enumerate(skeleton_nodes):
        print(f"  {i}: {bone}")
    print()
    
    # Print full hierarchy
    print("Full Hierarchy:")
    print("-" * 40)
    print_hierarchy(root)
    
    # Cleanup
    manager.Destroy()

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_fbx.py <path_to_fbx>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    
    analyze_fbx(filepath)

if __name__ == "__main__":
    main()
