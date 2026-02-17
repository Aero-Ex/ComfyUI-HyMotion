import bpy
import mathutils
import sys
import os

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_fbx(filepath, suffix):
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    
    # Import
    bpy.ops.import_scene.fbx(filepath=filepath)
    
    # Find the imported armature and rename it/its bones
    armature = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            armature = obj
            armature.name = f"{armature.name}_{suffix}"
            break
    
    return armature

def compare_transforms(t1, t2, threshold=0.001):
    # Returns True if they are basically the same
    diff_loc = (t1.to_translation() - t2.to_translation()).length
    # Comparison of rotations using quaternions
    q1 = t1.to_quaternion()
    q2 = t2.to_quaternion()
    diff_rot = q1.rotation_difference(q2).angle
    
    return diff_loc < threshold and diff_rot < threshold, diff_loc, diff_rot

def main():
    fbx1_path = "/home/aero/comfy/ComfyUI/output/hymotion_retarget/retarget_20260207_213457798_787a73b0_000.fbx"
    fbx2_path = "/home/aero/comfy/ComfyUI/output/hymotion_retarget/test_fixed_mhr_retarget.fbx"

    if not os.path.exists(fbx1_path):
        print(f"File not found: {fbx1_path}")
        return
    if not os.path.exists(fbx2_path):
        print(f"File not found: {fbx2_path}")
        return

    clear_scene()

    print(f"Importing {fbx1_path}...")
    arm1 = import_fbx(fbx1_path, "FBX1")
    
    print(f"Importing {fbx2_path}...")
    arm2 = import_fbx(fbx2_path, "FBX2")

    if not arm1 or not arm2:
        print("Failed to find armatures in one or both files.")
        return

    # Check frame range
    scene = bpy.context.scene
    start_frame = int(min(arm1.animation_data.action.frame_range[0], arm2.animation_data.action.frame_range[0]))
    end_frame = int(max(arm1.animation_data.action.frame_range[1], arm2.animation_data.action.frame_range[1]))

    print(f"Comparing frames {start_frame} to {end_frame}...")

    # Get common bones
    bones1 = {b.name: b for b in arm1.pose.bones}
    bones2 = {b.name: b for b in arm2.pose.bones}
    
    # Matching bones names (stripping suffix if any, but import_fbx renaming might need to be careful)
    # Actually import_fbx renames the Object, not the data/bones. So bone names should match.
    common_bone_names = set(bones1.keys()) & set(bones2.keys())
    
    if not common_bone_names:
        print("No common bones found between the two armatures.")
        print(f"Arm1 bones: {list(bones1.keys())[:5]}...")
        print(f"Arm2 bones: {list(bones2.keys())[:5]}...")
        return

    print(f"Found {len(common_bone_names)} common bones.")

    diff_count = 0
    max_diff_loc = 0
    max_diff_rot = 0

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        
        # We need to update the scene to calculate world matrices
        bpy.context.view_layer.update()
        
        frame_had_diff = False
        for bone_name in common_bone_names:
            b1 = arm1.pose.bones[bone_name]
            b2 = arm2.pose.bones[bone_name]
            
            # World space matrix
            # Note: arm.matrix_world @ bone.matrix
            m1 = arm1.matrix_world @ b1.matrix
            m2 = arm2.matrix_world @ b2.matrix
            
            same, d_loc, d_rot = compare_transforms(m1, m2)
            
            if not same:
                if not frame_had_diff:
                    print(f"\nFrame {frame}:")
                    frame_had_diff = True
                
                print(f"  Bone '{bone_name}': Loc Diff: {d_loc:.6f}, Rot Diff: {d_rot:.6f}")
                diff_count += 1
                max_diff_loc = max(max_diff_loc, d_loc)
                max_diff_rot = max(max_diff_rot, d_rot)

    print("\n" + "="*30)
    print("Comparison Summary:")
    print(f"Total differences found: {diff_count}")
    print(f"Max Location Difference: {max_diff_loc:.6f}")
    print(f"Max Rotation Difference: {max_diff_rot:.6f}")
    if diff_count == 0:
        print("FILES ARE IDENTICAL (within threshold)")
    else:
        print("FILES ARE DIFFERENT")
    print("="*30)

if __name__ == "__main__":
    main()
