import fbx
from fbx import FbxManager, FbxImporter, FbxScene, FbxIOSettings, FbxTime
import numpy as np
from scipy.spatial.transform import Rotation as R

def reverse_engineer(file_path, bone_name):
    manager = FbxManager.Create()
    ios = FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    importer = FbxImporter.Create(manager, "")
    importer.Initialize(file_path, -1, ios)
    scene = FbxScene.Create(manager, "")
    importer.Import(scene)
    importer.Destroy()

    node = scene.FindNodeByName(bone_name)
    if not node:
        print(f"Node {bone_name} not found")
        return

    print(f"\nReverse Engineering: {bone_name} in {file_path}")
    
    # 1. Properties
    t = node.LclTranslation.Get()
    r = node.LclRotation.Get()
    pre = node.PreRotation.Get()
    post = node.PostRotation.Get()
    print(f"  Lcl T: {t}")
    print(f"  Lcl R: {r}")
    print(f"  Pre R: {pre}")
    print(f"  PostR: {post}")

    # 2. Evaluations
    time = FbxTime()
    time.SetFrame(0)
    
    eval_local_r = node.EvaluateLocalRotation(time)
    print(f"  EvaluateLocalRotation(f0): {eval_local_r[0]:.3f}, {eval_local_r[1]:.3f}, {eval_local_r[2]:.3f}")
    
    eval_gt = node.EvaluateGlobalTransform(time)
    eval_gr = eval_gt.GetR()
    eval_gt_t = eval_gt.GetT()
    print(f"  EvaluateGlobalTransform(f0) R: {eval_gr[0]:.3f}, {eval_gr[1]:.3f}, {eval_gr[2]:.3f}")
    print(f"  EvaluateGlobalTransform(f0) T: {eval_gt_t[0]:.3f}, {eval_gt_t[1]:.3f}, {eval_gt_t[2]:.3f}")

    # 3. Parent Evaluation
    parent = node.GetParent()
    if parent:
        p_gt = parent.EvaluateGlobalTransform(time)
        p_gt_r = p_gt.GetR()
        p_gt_t = p_gt.GetT()
        print(f"  Parent ({parent.GetName()}) Wld R: {p_gt_r[0]:.3f}, {p_gt_r[1]:.3f}, {p_gt_r[2]:.3f}")
        print(f"  Parent ({parent.GetName()}) Wld T: {p_gt_t[0]:.3f}, {p_gt_t[1]:.3f}, {p_gt_t[2]:.3f}")

    # 4. Math verification
    # Try different combinations of Pre/Post
    pq = R.from_euler('xyz', [pre[0], pre[1], pre[2]], degrees=True)
    rq = R.from_euler('xyz', [r[0], r[1], r[2]], degrees=True)
    postq = R.from_euler('xyz', [post[0], post[1], post[2]], degrees=True)
    
    print("\n  Math Checks:")
    # Hypothesis 1: LocalEffective = Pre * Lcl * Post
    h1 = (pq * rq * postq).as_euler('xyz', degrees=True)
    print(f"    Pre * Lcl * Post: {h1.round(3)}")
    
    # Hypothesis 2: LocalEffective = Pre * Lcl * inv(Post)
    h2 = (pq * rq * postq.inv()).as_euler('xyz', degrees=True)
    print(f"    Pre * Lcl * inv(Post): {h2.round(3)}")
    
    # Hypothesis 3: ParentWorld * LocalEffective = World
    if parent:
        p_r = R.from_euler('xyz', [p_gt_r[0], p_gt_r[1], p_gt_r[2]], degrees=True)
        w_r = R.from_euler('xyz', [eval_gr[0], eval_gr[1], eval_gr[2]], degrees=True)
        # Expected LocalEffective
        eff_q = p_r.inv() * w_r
        eff_e = eff_q.as_euler('xyz', degrees=True)
        print(f"    Req. LocalEffective (inv(P)*W): {eff_e.round(3)}")

if __name__ == "__main__":
    print("--- SOURCE ---")
    reverse_engineer("/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx", "lThighBend")
    print("\n--- RETARGETED ---")
    reverse_engineer("/home/aero/Documents/G3.fbx", "lThighBend")
    
    print("\n--- HIP SOURCE ---")
    reverse_engineer("/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx", "hip")
    print("\n--- HIP RETARGETED ---")
    reverse_engineer("/home/aero/Documents/G3.fbx", "hip")
