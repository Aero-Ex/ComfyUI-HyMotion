import numpy as np
import os

def main():
    npz_path = "/home/aero/comfy/ComfyUI/output/hymotion_npz/wakingfromdie_20260123_182552595_c2e073c9_000.npz"
    if not os.path.exists(npz_path):
        print("File not found")
        return
        
    data = np.load(npz_path)
    kps = data['keypoints3d']
    transl = data['transl']
    
    print(f"NPZ: {os.path.basename(npz_path)}")
    print(f"Shape KPS: {kps.shape}, Transl: {transl.shape}")
    spine_idx = 3
    neck_idx = 12
    
    print(f"Frame 0 Spine-Neck Distance: {np.linalg.norm(kps[0, neck_idx] - kps[0, spine_idx]):.4f}m")
    
    # Check if this changes over time
    print(f"Frame End Spine-Neck Distance: {np.linalg.norm(kps[-1, neck_idx] - kps[-1, spine_idx]):.4f}m")

if __name__ == "__main__":
    main()
