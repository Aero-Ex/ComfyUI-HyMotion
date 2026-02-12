import subprocess
import sys
import os

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

def install():
    python_path = get_base_python_path()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")
    
    extra_index_url = "https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple"
    package_name = "fbxsdkpy==2020.1.post2"
    
    use_uv = is_uv_available()
    
    if use_uv:
        print("Using uv for installation...")
        # fbxsdkpy installation (explicitly splitting arguments)
        fbx_cmd = [python_path, "-m", "uv", "pip", "install", "--extra-index-url", extra_index_url, package_name]
        # rest of requirements
        req_cmd = [python_path, "-m", "uv", "pip", "install", "-r", requirements_path]
    else:
        print("Using pip for installation...")
        fbx_cmd = [python_path, "-m", "pip", "install", "--extra-index-url", extra_index_url, package_name]
        req_cmd = [python_path, "-m", "pip", "install", "-r", requirements_path]
    
    print(f"Installing {package_name}...")
    if run_command(fbx_cmd):
        print(f"Installing requirements from {requirements_path}...")
        run_command(req_cmd)
    else:
        print(f"Failed to install {package_name}. Skipping remaining requirements.")

if __name__ == "__main__":
    install()
