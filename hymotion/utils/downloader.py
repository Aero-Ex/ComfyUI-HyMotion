import os
import requests
from tqdm import tqdm
import folder_paths

MODEL_METADATA = {
    "HY-Motion-1.0 (Full)": {
        "url": "https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0/latest.ckpt",
        "dest": "hymotion/HY-Motion-1.0",
        "filename": "latest.ckpt"
    },
    "HY-Motion-1.0 (Lite)": {
        "url": "https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0-Lite/latest.ckpt",
        "dest": "hymotion/HY-Motion-1.0-Lite",
        "filename": "latest.ckpt"
    },
    "CLIP-ViT-Large-Patch14": {
        "url": "https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/text_encoders/clip-vit-large-patch14.safetensors",
        "dest": "text_encoders",
        "filename": "clip-vit-large-patch14.safetensors"
    },
    "Qwen3-8B (FP8)": {
        "url": "https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/text_encoders/Qwen3-8B_fp8.safetensors",
        "dest": "text_encoders",
        "filename": "Qwen3-8B_fp8.safetensors"
    }
}

def download_file(url, dest_path):
    """Download a file with progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    print(f"[HY-Motion] Downloading {os.path.basename(dest_path)}...")
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
    
    print(f"[HY-Motion] Download complete: {dest_path}")
    return dest_path

def get_model_path(model_name, custom_path=None):
    """Resolve the destination path for a model."""
    if model_name not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {model_name}")
    
    meta = MODEL_METADATA[model_name]
    
    if custom_path and custom_path.strip():
        base_path = custom_path
    else:
        # Default to ComfyUI models directory
        base_path = folder_paths.models_dir
        
    return os.path.join(base_path, meta["dest"], meta["filename"])
