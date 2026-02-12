import os
import sys

try:
    import fbx
except ImportError:
    fbx = None

def bake_transform_to_fbx(input_path, output_path, translation, rotation, scale):
    """
    Bakes static transformations into an FBX file.
    
    Args:
        input_path: Path to source FBX
        output_path: Path to save baked FBX
        translation: [x, y, z] in CM
        rotation: [x, y, z] in degrees
        scale: [x, y, z] relative
    """
    if fbx is None:
        raise ImportError("FBX SDK (fbx module) not found.")

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    # Load scene
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(input_path, -1, manager.GetIOSettings()):
        err = importer.GetStatus().GetErrorString()
        importer.Destroy()
        manager.Destroy()
        raise Exception(f"Failed to load FBX: {err}")

    scene = fbx.FbxScene.Create(manager, "MyScene")
    importer.Import(scene)
    importer.Destroy()

    # Find root node (the child of the actual invisible Scene root)
    root_node = scene.GetRootNode()
    
    # We apply to the first meaningful child (usually the model's 'root' or 'Pelvis')
    # If we apply to scene.GetRootNode() directly, it might not export correctly in some viewers
    target_node = None
    if root_node.GetChildCount() > 0:
        target_node = root_node.GetChild(0)
    else:
        target_node = root_node # Fallback

    print(f"Baking transform to target node: {target_node.GetName()}")

    # Apply translation
    target_node.LclTranslation.Set(fbx.FbxDouble3(translation[0], translation[1], translation[2]))
    
    # Apply rotation
    target_node.LclRotation.Set(fbx.FbxDouble3(rotation[0], rotation[1], rotation[2]))
    
    # Apply scale
    target_node.LclScaling.Set(fbx.FbxDouble3(scale[0], scale[1], scale[2]))

    # Save scene
    exporter = fbx.FbxExporter.Create(manager, "")
    if not exporter.Initialize(output_path, -1, manager.GetIOSettings()):
        err = exporter.GetStatus().GetErrorString()
        exporter.Destroy()
        manager.Destroy()
        raise Exception(f"Failed to save FBX: {err}")

    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
    
    return True

if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tx", type=float, default=0)
    parser.add_argument("--ty", type=float, default=0)
    parser.add_argument("--tz", type=float, default=0)
    parser.add_argument("--rx", type=float, default=0)
    parser.add_argument("--ry", type=float, default=0)
    parser.add_argument("--rz", type=float, default=0)
    parser.add_argument("--sx", type=float, default=1)
    parser.add_argument("--sy", type=float, default=1)
    parser.add_argument("--sz", type=float, default=1)
    
    args = parser.parse_args()
    bake_transform_to_fbx(args.input, args.output, [args.tx, args.ty, args.tz], [args.rx, args.ry, args.rz], [args.sx, args.sy, args.sz])
