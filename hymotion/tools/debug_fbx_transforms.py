import sys, os
import numpy as np
import fbx
from fbx import FbxManager, FbxImporter, FbxScene, FbxIOSettings, FbxTime

def debug_fbx(file_path):
    manager = FbxManager.Create()
    ios = FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    importer = FbxImporter.Create(manager, "")
    
    if not importer.Initialize(file_path, -1, ios):
        print(f"Failed to initialize importer: {importer.GetStatus().GetErrorString()}")
        return

    scene = FbxScene.Create(manager, "")
    importer.Import(scene)
    importer.Destroy()

    print(f"\nDebugging FBX: {file_path}")
    print("="*80)

    tmode = scene.GetGlobalSettings().GetTimeMode()

    def walk_nodes(node, depth=0):
        name = node.GetName()
        indent = "  " * depth
        
        # Get properties
        lcl_t = node.LclTranslation.Get()
        lcl_r = node.LclRotation.Get()
        pre_r = node.PreRotation.Get()
        post_r = node.PostRotation.Get()
        
        # Evaluate at frame 0
        time = FbxTime()
        time.SetFrame(0, tmode)
        eval_gt = node.EvaluateGlobalTransform(time)
        eval_t = eval_gt.GetT()
        eval_r = eval_gt.GetR()
        
        print(f"{indent}Node: {name}")
        print(f"{indent}  Parent: {node.GetParent().GetName() if node.GetParent() else 'None'}")
        print(f"{indent}  Lcl T: [{lcl_t[0]:.3f}, {lcl_t[1]:.3f}, {lcl_t[2]:.3f}]")
        print(f"{indent}  Lcl R: [{lcl_r[0]:.3f}, {lcl_r[1]:.3f}, {lcl_r[2]:.3f}]")
        print(f"{indent}  Pre R: [{pre_r[0]:.3f}, {pre_r[1]:.3f}, {pre_r[2]:.3f}]")
        print(f"{indent}  PostR: [{post_r[0]:.3f}, {post_r[1]:.3f}, {post_r[2]:.3f}]")
        print(f"{indent}  Wld T (f0): [{eval_t[0]:.3f}, {eval_t[1]:.3f}, {eval_t[2]:.3f}]")
        print(f"{indent}  Wld R (f0): [{eval_r[0]:.3f}, {eval_r[1]:.3f}, {eval_r[2]:.3f}]")
        
        for i in range(node.GetChildCount()):
            walk_nodes(node.GetChild(i), depth + 1)

    walk_nodes(scene.GetRootNode())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_fbx(sys.argv[1])
    else:
        # Default to the files we're working with
        print("Checking Source FBX:")
        debug_fbx("/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx")
        print("\nChecking Retargeted FBX:")
        debug_fbx("/home/aero/Documents/G3.fbx")
