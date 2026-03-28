import time
import torch
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

from text_to_token_online import load_inference_model, clean_tokens
from token_to_tracker_online import load_vae_model, decode_token_to_features, inv_rename_transform, compute_tracker_data_online
from feature_to_joints import recover_g1_motion


from tracker.scripts.mujoco_evaluate_view_online import (
    compute_observation, draw_ghost, MotionLoaderOnline,
    kps_base, kds_base, scale_base, default_angles, isaaclab_to_mujoco_reindex
)

class Config:
    xml_path = "./tracker/source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1_act_fixed.xml"
    policy_path = "./tracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx"
    vae_ckpt_path = "experiments/mgpt/VQVAE_UnitreeG1_v4/checkpoints/epoch=999-v1.ckpt"
    vae_config_path = "configs/config_ug1_stage1.yaml"
    meta_path = "datasets/unitreeg1/meta"
    
    power_scale = 1.25 # 1.2
    damp_scale = 0.8 # 0.5
    action_mult = 1.1 # 1.5
    stride = 1

def main_online():
    print("initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_kps = kps_base * Config.power_scale 
    current_kds = kds_base * Config.damp_scale  
    current_scale = scale_base * Config.action_mult
    
    # LLM 
    llm_model = load_inference_model() 
    
    # VAE 
    vae_model = load_vae_model(ckpt_path=Config.vae_ckpt_path, config_path=Config.vae_config_path, device=device)
    
    # RL Policy (ONNX)
    policy_session = ort.InferenceSession(Config.policy_path)
    obs_name = policy_session.get_inputs()[0].name
    
    # MuJoCo 
    mj_model = mujoco.MjModel.from_xml_path(Config.xml_path)
    mj_data = mujoco.MjData(mj_model)

    print("\n✅ Ready!")

    while True:
        text_prompt = input("\n>>> Inputs: (eg. 'A robot walks forward', q): ").strip()
        if text_prompt.lower() in ['q', 'exit']:
            break
        if not text_prompt: continue

        # STAGE 1: Text to Token 
        print(f"   ⏳ [1/3] LLM thinking...")
        with torch.no_grad():
            output_tokens, _ = llm_model.lm.generate_direct([text_prompt], do_sample=False, max_length=120)
            final_tokens = clean_tokens(output_tokens[0])

        # STAGE 2: Token to Tracker
        print(f"   ⏳ [2/3] VAE decoding...")
        features_tensor, mean, std = decode_token_to_features(vae_model, final_tokens, meta_path=Config.meta_path, device=device)
        
        with torch.no_grad():
            global_pos_t, global_rot_t = recover_g1_motion(features_tensor, mean=mean, std=std)
        
        global_pos = global_pos_t[0].cpu().numpy() 
        global_rot = global_rot_t[0].cpu().numpy() 
        rp_z, rr_z = inv_rename_transform(global_pos[:, 0, :], global_rot[:, 0, :])
        _, jg_z = inv_rename_transform(global_pos[:, 0, :], global_rot[:, 1:, :])
        
        motion_dict = compute_tracker_data_online(rp_z, rr_z, jg_z, Config.xml_path)

        # STAGE 3: MuJoCo Simulation
        print(f"   🚀 [3/3] Opening MuJoCo Viewer...")
        loader = MotionLoaderOnline(motion_dict) 
        
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.cam.lookat[:] = [0,0,0.8]; viewer.cam.distance = 3.0
            last_action = np.zeros(29, dtype=np.float32)

            start_q_mj = loader.joint_pos[0]
            start_root = loader.body_pos[0][0]
            start_quat = loader.body_ori[0][0]
            mj_data.qpos[7:] = start_q_mj
            mj_data.qpos[:3] = start_root + np.array([0,0,0.01])
            mj_data.qpos[3:7] = start_quat
            mj_data.qvel[:] = 0
            mujoco.mj_step(mj_model, mj_data)
            
            for t in range(loader.T):
                if not viewer.is_running(): 
                    break

                obs = compute_observation(mj_data, loader, t, last_action, stride=Config.stride)
                net_dim = policy_session.get_inputs()[0].shape[1]
                if obs.shape[0] < net_dim: 
                    obs = np.pad(obs, (0, net_dim - obs.shape[0]))
                
                action_isaac = policy_session.run(None, {obs_name: obs.reshape(1, -1)})[0].squeeze()
                last_action = action_isaac
                
                action_mj = action_isaac[isaaclab_to_mujoco_reindex]
                target = action_mj * current_scale + default_angles
                
                for _ in range(10):
                    mj_data.ctrl[:] = current_kps * (target - mj_data.qpos[7:]) - current_kds * mj_data.qvel[6:]
                    mujoco.mj_step(mj_model, mj_data)
                
                draw_ghost(viewer, loader, t)
                
                if mj_data.qpos[2] < 0.3:
                    print(f"   ⚠️ Falling at {t} ！")
                    break 
                
                viewer.sync()
                time.sleep(0.02)
        
        print("\n✅ finished.")

if __name__ == "__main__":
    main_online()