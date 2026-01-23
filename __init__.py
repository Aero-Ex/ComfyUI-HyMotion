import os
import folder_paths

from .nodes_modular import NODE_CLASS_MAPPINGS_MODULAR, NODE_DISPLAY_NAME_MAPPINGS_MODULAR
from .nodes_3d_viewer import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_3D, NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_3D
from .nodes_2d_preview import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_2D, NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_2D
from .nodes_extra import NODE_CLASS_MAPPINGS_EXTRA, NODE_DISPLAY_NAME_MAPPINGS_EXTRA

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Tell ComfyUI where to find our web extensions (CRITICAL for 3D viewer!)
WEB_DIRECTORY = os.path.join(CURRENT_DIR, "web")

# Register hymotion as a native ComfyUI model folder type
# This allows models to be stored in models/hymotion/ and be discovered automatically
folder_paths.folder_names_and_paths["hymotion"] = (
    [os.path.join(folder_paths.models_dir, "hymotion")],
    folder_paths.supported_pt_extensions  # Use standard extensions
)

# Add backward compatibility for old model location
# This ensures existing models in models/HY-Motion/ckpts/tencent/ are still discoverable
legacy_path = os.path.join(folder_paths.models_dir, "HY-Motion", "ckpts", "tencent")
if os.path.exists(legacy_path):
    folder_paths.add_model_folder_path("hymotion", legacy_path)

# Register text_encoders folder for GGUF and other models
folder_paths.folder_names_and_paths["hymotion_text_encoders"] = (
    [os.path.join(folder_paths.models_dir, "text_encoders")],
    {".gguf", ".bin", ".safetensors", ".pt", ".ckpt"}
)

# Register FBX templates folder
folder_paths.folder_names_and_paths["hymotion_fbx_templates"] = (
    [os.path.join(CURRENT_DIR, "assets", "wooden_models")],
    {".fbx"}
)

# Register NPZ folder (scans both input and output for convenience)
folder_paths.folder_names_and_paths["hymotion_npz"] = (
    [
        os.path.join(folder_paths.get_output_directory(), "hymotion_npz"),
        os.path.join(folder_paths.get_input_directory(), "hymotion_npz")
    ],
    {".npz"}
)

# Combine nodes mappings
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS_MODULAR, **NODE_CLASS_MAPPINGS_2D, **NODE_CLASS_MAPPINGS_3D, **NODE_CLASS_MAPPINGS_EXTRA}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS_MODULAR, **NODE_DISPLAY_NAME_MAPPINGS_2D, **NODE_DISPLAY_NAME_MAPPINGS_3D, **NODE_DISPLAY_NAME_MAPPINGS_EXTRA}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
