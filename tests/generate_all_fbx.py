import subprocess
import os

# Paths
PYTHON_EXE = r"D:\@home\aero\comfy\comfy-win\Scripts\python.exe"
SCRIPT_PATH = "retargetfbxnpzfull.py"
OUTPUT_DIR = r"D:\@home\aero\comfy\ComfyUI\custom_nodes\ComfyUI-HyMotion"

# Sources
NPZ_SOURCE = r"D:\@home\aero\comfy\ComfyUI\output\hymotion_npz\motion_20260119_145100221_f1a62b87_000.npz"
FBX_SOURCE = r"D:\@home\aero\comfy\ComfyUI\output\hymotion_fbx\motion_20260119_145059463_32cbd0a1_000.fbx"

# Targets
TARGETS = [
    ("tar1", r"D:\rigged_1768725943_articulationxl.fbx"),
    ("tar2", r"D:\rigged_1768669659_articulationxl.fbx"),
    ("tar3", r"D:\rigged_1768725759_articulationxl.fbx"),
    ("tar4", r"D:\rigged_1768725846_articulationxl.fbx"),
    ("tar5", r"D:\rigged_1768725894_articulationxl.fbx"),
    ("tar6", r"D:\rigged_1768725910_articulationxl.fbx"),
    ("tar7", r"D:\rigged_1768725926_articulationxl.fbx"),
    ("tar8", r"D:\Human_Croc.fbx"),
    ("tar9", r"D:\manny--unreal--engine-5\source\Manny_Unreal_Engine_5.fbx"),
    ("tar10", r"D:\woman-capoeira\source\femeie_1.fbx"),
    ("teddy", r"D:\SK_teddy.fbx"),
]

def run_retarget(source, target_path, output_name):
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cmd = [
        PYTHON_EXE, SCRIPT_PATH,
        "--source", source,
        "--target", target_path,
        "--output", output_path
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  SUCCESS: {output_name}")
    else:
        print(f"  FAILED: {output_name}")
        print(result.stderr)

def main():
    for label, path in TARGETS:
        # Generate NPZ version
        run_retarget(NPZ_SOURCE, path, f"{label}_npz.fbx")
        # Generate FBX version
        run_retarget(FBX_SOURCE, path, f"{label}_.fbx")

if __name__ == "__main__":
    main()
