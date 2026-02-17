import os
import sys
import fbx
from fbx import *

def print_hierarchy(node, depth=0):
    print("  " * depth + node.GetName())
    for i in range(node.GetChildCount()):
        print_hierarchy(node.GetChild(i), depth + 1)

def main():
    manager = FbxManager.Create()
    scene = FbxScene.Create(manager, "")
    importer = FbxImporter.Create(manager, "")
    
    fbx_path = "/home/aero/Documents/G3F-Tpose-With Locks.fbx"
    if not importer.Initialize(fbx_path, -1, manager.GetIOSettings()):
        print(f"Failed to load {fbx_path}")
        return
        
    importer.Import(scene)
    importer.Destroy()
    
    print(f"Hierarchy for {fbx_path}:")
    print_hierarchy(scene.GetRootNode())

if __name__ == "__main__":
    main()
