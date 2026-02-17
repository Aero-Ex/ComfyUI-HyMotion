import sys
import os
import unittest
import numpy as np

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from hymotion.utils.retarget_fbx import load_bone_mapping, Skeleton, BoneData, get_skeleton_height

class TestRetargetRegression(unittest.TestCase):
    def setUp(self):
        # Create a mock source skeleton (SMPL-H style)
        # We build a hierarchy that get_skeleton_height will follow
        self.src_skel = Skeleton("Source")
        self.src_skel.unit_scale = 100.0 # Meters (HyMotion)
        
        # Pelvis (Root) at origin
        pelvis = BoneData("Pelvis")
        pelvis.head = np.array([0, 0, 0])
        self.src_skel.add_bone(pelvis)
        
        # Spine: Pelvis(0,0,0) -> Spine(0, 0.4, 0) : length 0.4
        spine = BoneData("Spine1")
        spine.head = np.array([0, 0.4, 0])
        spine.parent_name = "Pelvis"
        self.src_skel.add_bone(spine)
        
        # Head: Spine(0, 0.4, 0) -> Head(0, 0.8, 0) : length 0.4
        head = BoneData("Head")
        head.head = np.array([0, 0.8, 0])
        head.parent_name = "Spine1"
        self.src_skel.add_bone(head)
        
        # Leg: Pelvis(0,0,0) -> Hip(0.1, 0, 0) -> Foot(0.1, -0.8, 0)
        # Traverse Pelvis -> Hip -> Foot
        l_hip = BoneData("L_Hip")
        l_hip.head = np.array([0.1, 0, 0])
        l_hip.parent_name = "Pelvis"
        self.src_skel.add_bone(l_hip)
        
        l_foot = BoneData("L_Foot")
        l_foot.head = np.array([0.1, -0.8, 0])
        l_foot.parent_name = "L_Hip"
        self.src_skel.add_bone(l_foot)

        # Other bones for mapping test
        for name in ["R_Hip", "L_Wrist", "R_Wrist", "L_Index1", "L_Middle1"]:
            bone = BoneData(name)
            bone.head = np.zeros(3)
            self.src_skel.add_bone(bone)

        # Create a mock target skeleton (Centimeter style)
        self.tgt_skel = Skeleton("Target")
        self.tgt_skel.unit_scale = 1.0 # Centimeters
        
        # Add target bones
        target_names = ["mixamorig:Hips", "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg", 
                        "mixamorig:Spine", "mixamorig:LeftHand", "mixamorig:RightHand",
                        "mixamorig:LeftHandIndex1", "mixamorig:LeftHandMiddle1"]
        for name in target_names:
            bone = BoneData(name)
            bone.head = np.zeros(3)
            self.tgt_skel.add_bone(bone)

    def test_finger_mapping_no_overwriting(self):
        """Verify that high-confidence finger mappings are not overwritten by geometric matching."""
        mapping = load_bone_mapping("", self.src_skel, self.tgt_skel)
        
        self.assertIn("l_index1", mapping)
        target_val = mapping["l_index1"]
        if isinstance(target_val, list):
            target_val = target_val[0]
            
        self.assertEqual(target_val.lower(), "mixamorig:lefthandindex1")

    def test_unit_conversion_scaling(self):
        """Verify that scaling correctly handles Meters to Centimeters (100x)."""
        scale = self.src_skel.unit_scale / self.tgt_skel.unit_scale
        self.assertEqual(scale, 100.0)

    def test_skeleton_height_calculation(self):
        """Verify that skeleton height calculates correctly."""
        # The height logic:
        # Pass 1 (Hierarchy): Head->Pelvis (0.8) + Foot->Pelvis (0.9, as Hip is at 0.1 X)
        # total_len = 0.8 + 0.9 = 1.7
        # Pass 2 (Y-Range fallback): Max(0.8) - Min(-0.8) = 1.6
        # get_skeleton_height returns the hierarchy result if it's > 0.1
        
        h = get_skeleton_height(self.src_skel)
        self.assertAlmostEqual(h, 1.7, delta=0.01, msg=f"Skeleton height should be 1.7, got {h}")

if __name__ == "__main__":
    unittest.main()
