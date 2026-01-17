import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import folder_paths
import comfy.utils

class HYMotionPreview2D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8, "tooltip": "Width of the rendered preview image in pixels."}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8, "tooltip": "Height of the rendered preview image in pixels."}),
                "up_axis": (["y", "z"], {"default": "y", "tooltip": "Select the vertical axis of the motion data. HY-Motion uses 'y'."}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Index of the sample to preview when multiple motions are generated in a batch."}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "The specific frame to display when frame_skip is set to 0 (single frame mode)."}),
                "frame_skip": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "0 for single frame preview. 1+ to render an animation sequence, skipping this many frames between renders."}),
                "keep_duration": ("BOOLEAN", {"default": False, "tooltip": "If enabled, duplicates rendered frames to maintain the original animation length/speed when skipping frames."}),
                "max_frames": ("INT", {"default": 100, "min": 1, "max": 500, "step": 1, "tooltip": "Maximum number of frames to render in animation mode to prevent memory issues."}),
                "draw_skeleton": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw the SMPL-H skeleton bones connecting the joints."}),
                "elevation": ("INT", {"default": 20, "min": -90, "max": 90, "step": 1, "tooltip": "Vertical camera angle in degrees. 0 is eye-level, 90 is bird's eye view."}),
                "azimuth": ("INT", {"default": 1, "min": -180, "max": 180, "step": 1, "tooltip": "Horizontal camera rotation in degrees. 0 is front view, 180 is back view."}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.01, "tooltip": "Zoom control. Values > 1.0 zoom out (character appears smaller), values < 1.0 zoom in."}),
            },
            "optional": {
                "motion_data": ("HYMOTION_DATA", {"tooltip": "Direct motion data input from a sampler node."}),
                "npz_path": ("STRING", {"default": "", "tooltip": "Optional path to an .npz file containing motion data to preview."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_preview"
    CATEGORY = "HY-Motion/view"

    def render_preview(self, width, height, up_axis, batch_index, frame_index, frame_skip, keep_duration, max_frames, draw_skeleton, elevation, azimuth, scale, motion_data=None, npz_path=""):
        def safe_float(v, default):
            try:
                if v is None or v == "": return default
                return float(v)
            except: return default
        
        def safe_int(v, default):
            try:
                if v is None or v == "": return default
                return int(float(v))
            except: return default

        elevation = safe_int(elevation, 20)
        azimuth = safe_int(azimuth, 1)
        scale = safe_float(scale, 0.5)
        
        width = safe_int(width, 512)
        height = safe_int(height, 512)
        batch_index = safe_int(batch_index, 0)
        frame_index = safe_int(frame_index, 0)
        frame_skip = safe_int(frame_skip, 1)
        max_frames = safe_int(max_frames, 100)

        keypoints = None
        
        # 1. Try loading from motion_data
        if motion_data is not None:
            if "keypoints3d" in motion_data.output_dict:
                # Shape: (B, T, J, 3)
                k3d = motion_data.output_dict["keypoints3d"]
                # Clamp batch index
                b = min(batch_index, k3d.shape[0] - 1)
                keypoints = k3d[b].detach().cpu().numpy()
        
        # 2. Try loading from npz_path if motion_data is missing
        if keypoints is None and npz_path.strip():
            full_path = npz_path.strip()
            if not os.path.isabs(full_path):
                # Try input and output directories
                p_in = os.path.join(folder_paths.get_input_directory(), full_path)
                p_out = os.path.join(folder_paths.get_output_directory(), full_path)
                if os.path.exists(p_in):
                    full_path = p_in
                elif os.path.exists(p_out):
                    full_path = p_out
            
            if os.path.exists(full_path):
                try:
                    data = np.load(full_path)
                    # Common keys: 'keypoints3d', 'motion', 'pos'
                    for key in ['keypoints3d', 'motion', 'pos', 'joints']:
                        if key in data:
                            keypoints = data[key]
                            break
                    if keypoints is None:
                        # Try first key if none of the above match
                        keypoints = data[data.files[0]]
                except Exception as e:
                    print(f"[HY-Motion] Error loading NPZ: {e}")

        if keypoints is None:
            # Return blank image if no data found
            return (torch.zeros((1, height, width, 3)),)

        # Ensure keypoints is (T, J, 3)
        if len(keypoints.shape) == 4: # (B, T, J, 3)
            # Clamp batch index
            b = min(batch_index, keypoints.shape[0] - 1)
            keypoints = keypoints[b]
        
        if len(keypoints.shape) == 2:
            if keypoints.shape[1] == 3:
                # Case: (J, 3) -> Single frame, add time dimension
                keypoints = keypoints[np.newaxis, :, :]
            else:
                # Case: (T, J*3) -> Flattened joints, reshape to (T, J, 3)
                keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
        
        # Final check: if it's still not 3D, we can't process it
        if len(keypoints.shape) != 3:
            print(f"[HY-Motion] Error: Unexpected keypoints shape {keypoints.shape}")
            return (torch.zeros((1, height, width, 3)),)
        
        # Coordinate system remapping
        if up_axis == "y":
            # Swap Y and Z: (x, y, z) -> (x, z, y)
            keypoints = keypoints[:, :, [0, 2, 1]]
        
        num_frames = keypoints.shape[0]
        num_joints = keypoints.shape[1]
        
        # Determine frames to render
        if frame_skip > 0:
            indices = list(range(0, num_frames, frame_skip))
            if len(indices) > max_frames:
                print(f"[HY-Motion] Warning: Animation exceeds max_frames ({max_frames}). Truncating.")
                indices = indices[:max_frames]
        else:
            indices = [min(frame_index, num_frames - 1)]

        # Define skeleton (SMPL-H 22 joints)
        skeleton = []
        if draw_skeleton:
            if num_joints >= 22:
                skeleton = [
                    (0, 1), (0, 2), (0, 3), # Pelvis to L/R Hip, Spine1
                    (1, 4), (2, 5), (3, 6), # Hip to Knee, Spine2
                    (4, 7), (5, 8), (6, 9), # Knee to Ankle, Spine3
                    (7, 10), (8, 11), (9, 12), # Ankle to Foot, Neck
                    (12, 13), (12, 14), # Neck to L/R Shoulder
                    (12, 15), # Neck to Head
                    (13, 16), (14, 17), # Shoulder to Elbow
                    (16, 18), (17, 19), # Elbow to Wrist
                    (18, 20), (19, 21), # Wrist to L/R Hand
                ]

        # Prepare rendering
        images = []
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        
        # Calculate global bounds for consistent scaling across animation
        min_vals = keypoints.min(axis=(0, 1))
        max_vals = keypoints.max(axis=(0, 1))
        center = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() / (2.0 / scale)

        pbar = comfy.utils.ProgressBar(len(indices))

        for idx in indices:
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_axis_off()
            ax.view_init(elev=elevation, azim=azimuth)
            ax.set_xlim3d([center[0] - max_range, center[0] + max_range])
            ax.set_ylim3d([center[1] - max_range, center[1] + max_range])
            ax.set_zlim3d([center[2] - max_range, center[2] + max_range])
            
            joints = keypoints[idx]
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='blue', s=20)
            
            if draw_skeleton:
                for j1, j2 in skeleton:
                    if j1 < num_joints and j2 < num_joints:
                        ax.plot([joints[j1, 0], joints[j2, 0]], 
                                [joints[j1, 1], joints[j2, 1]], 
                                [joints[j1, 2], joints[j2, 2]], c='red', linewidth=2)

            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3]
            img_tensor = torch.from_numpy(img).float() / 255.0
            
            # Implementation of keep_duration: duplicate the frame if skip > 1
            if keep_duration and frame_skip > 1:
                for _ in range(frame_skip):
                    images.append(img_tensor)
            else:
                images.append(img_tensor)
            
            pbar.update(1)
            
        plt.close(fig)
        
        # Final safety check for keep_duration: if we exceeded original num_frames, truncate
        if keep_duration and frame_skip > 1:
            images = images[:num_frames]
            
        return (torch.stack(images),)

NODE_CLASS_MAPPINGS = {
    "HYMotionPreview2D": HYMotionPreview2D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionPreview2D": "HY-Motion 2D Motion Preview",
}
