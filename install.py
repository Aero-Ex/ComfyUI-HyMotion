import subprocess
import sys
import os
import platform

def run_command(command):
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    return True

def get_base_python_path():
    return sys.executable

def is_uv_available():
    try:
        subprocess.check_call([get_base_python_path(), "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def get_fbx_wheel_url():
    """Returns the direct download URL for the fbxsdkpy wheel based on platform and python version."""
    py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    is_windows = platform.system().lower() == "windows"
    
    # Mapping based on Inria GitLab registry (Project ID: 18692, Package ID: 6650)
    # Format: { (is_windows, py_ver): (file_id, filename) }
    wheel_map = {
        (True, "39"):  ("116752", "fbxsdkpy-2020.1.post2-cp39-none-win_amd64.whl"),
        (True, "310"): ("116749", "fbxsdkpy-2020.1.post2-cp310-none-win_amd64.whl"),
        (True, "311"): ("116750", "fbxsdkpy-2020.1.post2-cp311-none-win_amd64.whl"),
        (True, "312"): ("116751", "fbxsdkpy-2020.1.post2-cp312-none-win_amd64.whl"),
        (True, "313"): ("775987", "fbxsdkpy-2020.1.post2-cp313-none-win_amd64.whl"),
        (True, "314"): ("775988", "fbxsdkpy-2020.1.post2-cp314-none-win_amd64.whl"),
        
        (False, "39"):  ("776002", "fbxsdkpy-2020.1.post2-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        (False, "310"): ("116748", "fbxsdkpy-2020.1.post2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        (False, "311"): ("116755", "fbxsdkpy-2020.1.post2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        (False, "312"): ("116756", "fbxsdkpy-2020.1.post2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        (False, "313"): ("776014", "fbxsdkpy-2020.1.post2-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        (False, "314"): ("776016", "fbxsdkpy-2020.1.post2-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
    }
    
    file_data = wheel_map.get((is_windows, py_ver))
    if file_data:
        file_id, file_name = file_data
        return f"https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/files/{file_id}/{file_name}"
    return None

def download_assets():
    print("Downloading assets from Hugging Face...")
    try:
        from huggingface_hub import snapshot_download
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        snapshot_download(
            repo_id="Aero-Ex/Hy-Motion1.0",
            allow_patterns=["assets/*"],
            local_dir=current_dir,
            local_dir_use_symlinks=False
        )
        print("Assets downloaded successfully.")
    except ImportError:
        print("Error: huggingface_hub not installed. Cannot download assets.")
    except Exception as e:
        print(f"Error downloading assets: {e}")

def install():
    python_path = get_base_python_path()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")
    
    extra_index_url = "https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple"
    package_name = "fbxsdkpy>=2020.1"
    
    use_uv = is_uv_available()
    pip_cmd = ["uv", "pip"] if use_uv else ["pip"]
    
    print(f"Attempting standard installation of {package_name}...")
    fbx_cmd = [python_path, "-m"] + pip_cmd + ["install", "--extra-index-url", extra_index_url, package_name]
    
    if not run_command(fbx_cmd):
        print("Standard installation failed. Attempting robust direct wheel installation...")
        wheel_url = get_fbx_wheel_url()
        if wheel_url:
            print(f"Installing from direct URL: {wheel_url}")
            fallback_cmd = [python_path, "-m"] + pip_cmd + ["install", wheel_url]
            if not run_command(fallback_cmd):
                print(f"CRITICAL: Failed to install fbxsdkpy via direct wheel. Python version: {sys.version}")
        else:
            print(f"ERROR: No pre-built wheel found for this environment (Platform: {platform.system()}, Python: {sys.version.split()[0]})")

    print(f"Installing remaining requirements from {requirements_path}...")
    req_cmd = [python_path, "-m"] + pip_cmd + ["install", "-r", requirements_path]
    if run_command(req_cmd):
        download_assets()

if __name__ == "__main__":
    install()
