"""
Batch test script to analyze all target FBX files and compare mapping results.
Usage: python batch_test.py
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

# Import the retargeting module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retargetfbxnpzfull import load_fbx, load_npz, load_bone_mapping, Skeleton

# Target FBX files to test
TARGET_FILES = [
    r"D:\rigged_1768725943_articulationxl.fbx",  # tar1
    r"D:\rigged_1768725926_articulationxl.fbx",  # tar2 (FAILED)
    r"D:\rigged_1768725910_articulationxl.fbx",  # tar3
    r"D:\rigged_1768725894_articulationxl.fbx",  # tar4
    r"D:\rigged_1768725846_articulationxl.fbx",  # tar5
    r"D:\rigged_1768725759_articulationxl.fbx",  # tar6
    r"D:\Human_Croc.fbx",                         # tar7
    r"D:\SK_tira.fbx",                            # SK_tira
    r"D:\SK_teddy.fbx",                           # SK_teddy
]

# Source file (HyMotion output)
SOURCE_FILE = r"D:\@home\aero\comfy\ComfyUI\output\hymotion_fbx\motion_20260118_091426163_89203df0_002.fbx"

def get_bone_info(node, depth=0, parent_name=None, bones=None):
    """Recursively collect bone information."""
    if bones is None:
        bones = []
    
    name = node.GetName()
    attr = node.GetNodeAttribute()
    
    # Check if it's a skeleton bone (use type ID 2 for skeleton)
    is_skeleton = False
    if attr:
        try:
            type_id = attr.GetAttributeType()
            # Skeleton type is typically 2
            is_skeleton = (type_id == 2)
        except:
            pass
    
    if is_skeleton or "bone" in name.lower() or "armature" in name.lower():
        try:
            mat = node.EvaluateGlobalTransform()
            pos = [mat.GetT()[0], mat.GetT()[1], mat.GetT()[2]]
        except:
            pos = [0, 0, 0]
        
        children = [node.GetChild(i).GetName() for i in range(node.GetChildCount())]
        bones.append({
            'name': name,
            'depth': depth,
            'parent': parent_name,
            'position': pos,
            'children': children,
        })
    
    for i in range(node.GetChildCount()):
        get_bone_info(node.GetChild(i), depth + 1, name, bones)
    
    return bones

def analyze_fbx(filepath):
    """Analyze an FBX file and return detailed info."""
    if not os.path.exists(filepath):
        return {'error': f"File not found: {filepath}"}
    
    manager = FbxManager.Create()
    ios = FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    
    importer = FbxImporter.Create(manager, "")
    if not importer.Initialize(filepath, -1, manager.GetIOSettings()):
        return {'error': f"Failed to load: {importer.GetStatus().GetErrorString()}"}
    
    scene = FbxScene.Create(manager, "scene")
    importer.Import(scene)
    importer.Destroy()
    
    bones = get_bone_info(scene.GetRootNode())
    
    has_bone_0 = any("bone_0" == b['name'].lower() for b in bones)
    has_armature = any("armature" in b['name'].lower() for b in bones)
    
    # Find hierarchy issues
    bones_without_children = [b['name'] for b in bones if not b['children'] and b['depth'] < 3]
    
    manager.Destroy()
    
    return {
        'filename': os.path.basename(filepath),
        'total_bones': len(bones),
        'has_bone_0': has_bone_0,
        'has_armature': has_armature,
        'bones': bones,
        'bones_without_children': bones_without_children,
    }

def test_mapping(src_skel, tgt_filepath):
    """Test the bone mapping for a target file."""
    try:
        tgt_man, tgt_scene, tgt_skel = load_fbx(tgt_filepath)
        mapping = load_bone_mapping("", src_skel, tgt_skel)
        return {
            'success': True,
            'mapping_count': len(mapping),
            'mapping': dict(mapping),
            'source_bones': list(src_skel.bones.keys()),
            'target_bones': list(tgt_skel.bones.keys()),
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }

def main():
    print("=" * 80)
    print("BATCH FBX ANALYSIS AND MAPPING TEST")
    print("=" * 80)
    
    # Load source skeleton
    print(f"\nLoading source: {os.path.basename(SOURCE_FILE)}")
    if SOURCE_FILE.lower().endswith('.npz'):
        src_skel = load_npz(SOURCE_FILE)
    else:
        src_man, src_scene, src_skel = load_fbx(SOURCE_FILE, sample_rest_frame=0)
    print(f"Source has {len(src_skel.bones)} bones")
    
    # Process each target
    results = []
    for tgt_file in TARGET_FILES:
        print(f"\n{'='*60}")
        print(f"Target: {os.path.basename(tgt_file)}")
        print("-" * 60)
        
        # Analyze FBX structure
        info = analyze_fbx(tgt_file)
        if 'error' in info:
            print(f"  ERROR: {info['error']}")
            results.append({'file': tgt_file, 'error': info['error']})
            continue
        
        print(f"  Total bones: {info['total_bones']}")
        print(f"  Has bone_0: {info['has_bone_0']}")
        print(f"  Has Armature: {info['has_armature']}")
        
        if info['bones_without_children']:
            print(f"  Bones without children (depth<3): {info['bones_without_children'][:5]}...")
        
        # Test mapping
        mapping_result = test_mapping(src_skel, tgt_file)
        if mapping_result['success']:
            print(f"  Mapping count: {mapping_result['mapping_count']}")
            
            # Categorize mapped bones
            body_bones = ['pelvis', 'spine', 'neck', 'head', 'collar', 'shoulder', 'elbow', 'wrist']
            leg_bones = ['hip', 'knee', 'ankle', 'foot']
            finger_bones = ['thumb', 'index', 'middle', 'ring', 'pinky']
            
            mapped = mapping_result['mapping']
            body_count = sum(1 for k in mapped if any(b in k for b in body_bones))
            leg_count = sum(1 for k in mapped if any(b in k for b in leg_bones))
            finger_count = sum(1 for k in mapped if any(b in k for b in finger_bones))
            
            print(f"  Body bones mapped: {body_count}")
            print(f"  Leg bones mapped: {leg_count}")
            print(f"  Finger bones mapped: {finger_count}")
            
            # Show full mapping
            print(f"  Full mapping:")
            for src, tgt in sorted(mapped.items()):
                print(f"    {src} -> {tgt}")
        else:
            print(f"  MAPPING FAILED: {mapping_result['error']}")
        
        results.append({
            'file': os.path.basename(tgt_file),
            'info': info,
            'mapping': mapping_result,
        })
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'File':<40} {'Bones':<8} {'bone_0':<8} {'Mapped':<8} {'Status'}")
    print("-" * 80)
    
    for r in results:
        if 'error' in r:
            print(f"{os.path.basename(r['file']):<40} {'N/A':<8} {'N/A':<8} {'N/A':<8} ERROR")
        else:
            mapped = r['mapping']['mapping_count'] if r['mapping']['success'] else 0
            status = "OK" if mapped >= 15 else "POOR" if mapped > 0 else "FAILED"
            print(f"{r['file']:<40} {r['info']['total_bones']:<8} {str(r['info']['has_bone_0']):<8} {mapped:<8} {status}")

if __name__ == "__main__":
    main()
