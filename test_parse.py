import json

ROKOKO_CUSTOM_TO_SMPL = {
    "hip": "Pelvis",
    "leftUpLeg": "L_Hip",
    "leftLeg": "L_Knee",
    "leftFoot": "L_Ankle",
    "custom_bone_l_foot": "L_Foot",
    "rightUpLeg": "R_Hip",
    "rightLeg": "R_Knee",
    "rightFoot": "R_Ankle",
    "custom_bone_r_foot": "R_Foot",
    "custom_bone_spine1": "Spine1",
    "custom_bone_spine2": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "leftShoulder": "L_Collar",
    "custom_bone_l_shoulder": "L_Shoulder",
    "leftLowerArm": "L_Elbow",
    "leftHand": "L_Wrist",
    "rightShoulder": "R_Collar",
    "custom_bone_r_shoulder": "R_Shoulder",
    "rightLowerArm": "R_Elbow",
    "rightHand": "R_Wrist",
    "leftUpperArm": "L_Shoulder",
    "rightUpperArm": "R_Shoulder",
}

with open('/home/aero/comfy/ComfyUI/custom_nodes/ComfyUI-HyMotion/hymotion/skintokenmap.json', 'r') as f:
    data = json.load(f)

raw_bones = data.get("bones", {})
json_bones = {}
for k, v in raw_bones.items():
    if k in ROKOKO_CUSTOM_TO_SMPL:
        smpl_key = ROKOKO_CUSTOM_TO_SMPL[k]
        json_bones[smpl_key] = v

print("json_bones:")
for k, v in sorted(json_bones.items()):
    print(f"  {k}: {v}")
