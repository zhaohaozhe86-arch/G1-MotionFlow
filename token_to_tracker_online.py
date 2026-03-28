import torch
import numpy as np
import mujoco
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf
from os.path import join as pjoin

try:
    from feature_to_joints import recover_g1_motion
except ImportError:
    print("❌ Could not find feature_to_joints.py!")
    sys.exit(1)

try:
    from mGPT.archs.mgpt_vq import VQVae
except ImportError:
    print("⚠️  Warning: NO mGPT")

BODY_NAME_MAP_INV = {
    1: 'left_hip_pitch_link', 2: 'left_hip_roll_link', 3: 'left_hip_yaw_link',
    4: 'left_knee_link', 5: 'left_ankle_pitch_link', 6: 'left_ankle_roll_link',
    7: 'right_hip_pitch_link', 8: 'right_hip_roll_link', 9: 'right_hip_yaw_link',
    10: 'right_knee_link', 11: 'right_ankle_pitch_link', 12: 'right_ankle_roll_link',
    13: 'waist_yaw_link', 14: 'waist_roll_link', 15: 'waist_pitch_link',
    16: 'left_shoulder_pitch_link', 17: 'left_shoulder_roll_link', 18: 'left_shoulder_yaw_link',
    19: 'left_elbow_link', 20: 'left_wrist_roll_link', 21: 'left_wrist_pitch_link', 22: 'left_wrist_yaw_link',
    23: 'right_shoulder_pitch_link', 24: 'right_shoulder_roll_link', 25: 'right_shoulder_yaw_link',
    26: 'right_elbow_link', 27: 'right_wrist_roll_link', 28: 'right_wrist_pitch_link', 29: 'right_wrist_yaw_link'
}

def qinv_np(q):
    a = q.copy()
    a[..., 1:] *= -1
    return a

def qmul_np(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=-1)

def wxyz_to_xyzw(q): return q[..., [1, 2, 3, 0]]
def xyzw_to_wxyz(q): return q[..., [3, 0, 1, 2]]

def inv_rename_transform(pos, rot_wxyz):
    pos_zup = pos[..., [2, 0, 1]]
    rot_xyzw = wxyz_to_xyzw(rot_wxyz)
    rot_xyzw_zup = rot_xyzw[..., [2, 0, 1, 3]]
    rot_wxyz_zup = xyzw_to_wxyz(rot_xyzw_zup)
    return pos_zup, rot_wxyz_zup

def load_vae_model(ckpt_path, config_path, device):
    base_cfg = OmegaConf.load(config_path)
    try:
        vae_params = base_cfg.model.params.motion_vae.default.params
        vae_params = OmegaConf.to_container(vae_params, resolve=False)
    except Exception:
        vae_params = {
            "code_num": 512, "code_dim": 512, "output_emb_width": 512,
            "down_t": 2, "stride_t": 2, "width": 512, "depth": 3,
            "dilation_growth_rate": 3, "activation": 'relu'
        }

    vae_params['nfeats'] = 363
    vae_params.pop('ablation', None)
    vae_params.pop('norm', None)

    model = VQVae(**vae_params)
    print(f"Loading VAE from {ckpt_path}...")
    import torch.serialization
    _original_load = torch.load
    torch.load = lambda *a, **kw: _original_load(*a, **{**kw, 'weights_only': False})
    torch.serialization.load = torch.load
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_sd = {k.replace('motion_vae.', '').replace('vae.', ''): v for k, v in state_dict.items() 
              if 'motion_vae' in k or 'vae' in k or 'encoder' in k or 'decoder' in k}
    model.load_state_dict(new_sd, strict=False)
    model.eval().to(device)
    return model

def decode_token_to_features(model, tokens, meta_path, device):
    from os.path import join as pjoin
    import numpy as np
    import torch

    mean = np.load(pjoin(meta_path, "Mean.npy"))
    std = np.load(pjoin(meta_path, "Std.npy"))

    if isinstance(tokens, str):
        tokens = np.load(tokens)
        
    if tokens.ndim == 0: 
        tokens = tokens[None]
        
    tok_tensor = torch.tensor(tokens).long().to(device)
    if tok_tensor.dim() == 1: 
        tok_tensor = tok_tensor.unsqueeze(0)
    
    if hasattr(model.quantizer, 'codebook'): codebook = model.quantizer.codebook
    elif hasattr(model.quantizer, 'embedding'): codebook = model.quantizer.embedding.weight
    else: codebook = getattr(model.quantizer, '_codebook', None)
        
    z = torch.nn.functional.embedding(tok_tensor, codebook).permute(0, 2, 1)
    with torch.no_grad(): 
        out = model.decoder(z)
    
    out_tensor = out.permute(0, 2, 1) # (1, T, 363)
    return out_tensor, mean, std

def compute_tracker_data_online(root_pos, root_rot, joints_global_q, xml_path):
    print(f"Loading Tracker XML for kinematics: {xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print("\n❌ XML Loading failed！")
        return

    T = root_pos.shape[0]
    joint_angles = np.zeros((T, 29))
    
    print("Computing Joint Angles (Inverse FK)...")
    for i in range(29):
        body_idx = i + 1
        if body_idx not in BODY_NAME_MAP_INV: continue
        body_name = BODY_NAME_MAP_INV[body_idx]
        
        mj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if mj_body_id == -1: continue
        
        jnt_addr = model.body_jntadr[mj_body_id]
        if jnt_addr == -1: continue 
        jnt_axis = model.jnt_axis[jnt_addr] 
        
        parent_id = model.body_parentid[mj_body_id]
        q_child = joints_global_q[:, i, :]
        
        if parent_id == 0: 
            q_parent = root_rot
        else:
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
            p_idx = next((k - 1 for k, v in BODY_NAME_MAP_INV.items() if v == parent_name), -1)
            q_parent = joints_global_q[:, p_idx, :] if p_idx != -1 else root_rot 
        
        q_parent_inv = qinv_np(q_parent)
        q_rel = qmul_np(q_parent_inv, q_child)
        
        r = R.from_quat(wxyz_to_xyzw(q_rel))
        rot_vec = r.as_rotvec()
        angle = np.sum(rot_vec * jnt_axis, axis=1)

        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        
        joint_angles[:, i] = angle

    qpos = np.concatenate([root_pos, root_rot, joint_angles], axis=-1)
    
    print("Upsampling 20Hz -> 60Hz...")
    import scipy.interpolate
    ratio = 3
    T_new = (T - 1) * ratio + 1
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, T_new)
    
    qpos_60 = np.zeros((T_new, 36))
    for i in list(range(3)) + list(range(7, 36)):
        f = scipy.interpolate.interp1d(t_old, qpos[:, i], kind='linear')
        qpos_60[:, i] = f(t_new)
        
    key_rots = R.from_quat(wxyz_to_xyzw(qpos[:, 3:7]))
    slerp = scipy.spatial.transform.Slerp(t_old, key_rots)
    rot_new_xyzw = slerp(t_new).as_quat()
    qpos_60[:, 3:7] = xyzw_to_wxyz(rot_new_xyzw)
    
    data = mujoco.MjData(model)
    ref_global_pos = np.zeros((T_new, model.nbody-1, 3)) 
    ref_global_rot = np.zeros((T_new, model.nbody-1, 4))
    
    for t in range(T_new):
        data.qpos[:] = qpos_60[t]
        mujoco.mj_kinematics(model, data)
        ref_global_pos[t] = data.xpos[1:].copy()
        ref_global_rot[t] = data.xquat[1:].copy()
        
    dt = 1.0 / 60.0
    qvel = np.zeros((T_new, 35))
    qvel[:, 0:3] = np.gradient(qpos_60[:, 0:3], dt, axis=0) 
    qvel[:, 6:] = np.gradient(qpos_60[:, 7:], dt, axis=0)   
    
    global_vel = np.gradient(ref_global_pos, dt, axis=0)
    
    save_dict = {
        "qpos": qpos_60,
        "qvel": qvel,
        "ref_dof_pos": qpos_60[:, 7:],
        "ref_global_translation": ref_global_pos,
        "ref_global_rotation_quat": ref_global_rot,
        "ref_global_velocity": global_vel,
        "fps": 60.0
    }
    
    return save_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--ckpt", default="experiments/mgpt/VQVAE_UnitreeG1_v4/checkpoints/epoch=999-v1.ckpt")
    parser.add_argument("--config", default="configs/config_ug1_stage1.yaml")
    parser.add_argument("--meta", default="datasets/unitreeg1/meta")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== Starting motion decode ===")
    
    print("1. Decode Token -> Features Tensor (363 dims)...")
    model = load_vae_model(args.ckpt, args.config, device)
    features_tensor, mean, std = decode_token_to_features(model, args.token, args.meta, device)
    
    print("2. Calculationg with feature_to_joints.py ...")
    with torch.no_grad():
        global_pos_t, global_rot_t = recover_g1_motion(features_tensor, mean=mean, std=std)
    
    global_pos = global_pos_t[0].cpu().numpy() # (T, 30, 3)
    global_rot = global_rot_t[0].cpu().numpy() # (T, 30, 4) wxyz
    
    rp_y = global_pos[:, 0, :]               # Root Pos (T, 3)
    rr_y = global_rot[:, 0, :]               # Root Rot (T, 4)
    jg_y = global_rot[:, 1:, :]              # 子关节 Rot (T, 29, 4)
    
    print("3. Y-up -> Z-up...")
    rp_z, rr_z = inv_rename_transform(rp_y, rr_y)
    _, jg_z = inv_rename_transform(rp_y, jg_y)
    
    print("4. Calculating Tracker physics data...")
    compute_tracker_data_online(rp_z, rr_z, jg_z, args.xml, args.output)

if __name__ == "__main__":
    main()

'''
python token_to_tracker_363.py --token results/generated_tokens_npy/a_man_is_trying_to_k.npy \
--output results/generated_motion/a_man_is_trying_to_k.npz \
--xml tracker/source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1_act_fixed.xml
'''