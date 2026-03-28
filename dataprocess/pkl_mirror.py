import os
import joblib
import numpy as np
from tqdm import tqdm

SRC_DIR = r'./g1_data_pkl'  
OUT_DIR = r'./g1_data_pkl_mirrored'
MIRROR_AXIS_DIM = 1  # Y mirror

# ================= G1 Mirror Config =================
# 1. Permutation: 
G1_MIRROR_PERM = np.array([
    # Left Leg -> Right Leg
    6, 7, 8, 9, 10, 11,
    # Right Leg -> Left Leg
    0, 1, 2, 3, 4, 5,
    # Waist
    12, 13, 14,
    # Left Arm -> Right Arm
    22, 23, 24, 25, 26, 27, 28,
    # Right Arm -> Left Arm
    15, 16, 17, 18, 19, 20, 21
])

# 2. Signs: Roll/Yaw 
G1_MIRROR_SIGNS = np.array([
    # Left Leg
    1, -1, -1, 1, 1, -1, 
    # Right Leg
    1, -1, -1, 1, 1, -1,
    # Waist
    -1, -1, 1,
    # Left Arm
    1, -1, -1, 1, -1, 1, -1,
    # Right Arm
    1, -1, -1, 1, -1, 1, -1
])

def mirror_position(pos, axis_idx):
    new_pos = pos.copy()
    if pos.ndim == 1:
        new_pos[axis_idx] *= -1
    else:
        new_pos[:, axis_idx] *= -1
    return new_pos

def mirror_quaternion(quat, axis_idx):
    q = quat.copy()
    if axis_idx == 1: # Y Mirror
        if q.ndim == 1: q[[1, 3]] *= -1
        else: q[:, [1, 3]] *= -1
    elif axis_idx == 0: # X Mirror
        if q.ndim == 1: q[[0, 3]] *= -1
        else: q[:, [0, 3]] *= -1
    return q

def find_data_dict_recursive(obj, path=""):
    target_keys = {'dof', 'dof_pos', 'root_trans', 'root_pos', 'trans', 'pose_aa', 'qpos'}
    
    if isinstance(obj, dict):
        current_keys = set(obj.keys())
        if ('dof' in current_keys or 'dof_pos' in current_keys) or \
           len(current_keys.intersection(target_keys)) >= 2:
            return obj, path
        
        for k, v in obj.items():
            if isinstance(v, dict):
                found, found_path = find_data_dict_recursive(v, path + f"['{k}']")
                if found is not None:
                    return found, found_path
                    
    return None, None

def process_single_pkl(src_path, dst_path):
    try:
        raw_data = joblib.load(src_path)
        
        data_inner, trace_path = find_data_dict_recursive(raw_data)
        
        if data_inner is None:
            keys = list(raw_data.keys()) if isinstance(raw_data, dict) else "Not a dict"
            print(f"⚠️ skip {src_path}: no dof/root_pos. Top keys: {keys}")
            return
        
        final_data = joblib.load(src_path) 
        target_dict, _ = find_data_dict_recursive(final_data) 
        
        # 1. Process Dof
        dof_key = 'dof' if 'dof' in target_dict else 'dof_pos'
        if dof_key in target_dict:
            original_dof = target_dict[dof_key]
            
            is_expanded = False
            if original_dof.ndim == 3:
                original_dof = original_dof.squeeze(-1)
                is_expanded = True
            
            if original_dof.shape[1] == len(G1_MIRROR_PERM):
                new_dof = original_dof[:, G1_MIRROR_PERM] * G1_MIRROR_SIGNS
                if is_expanded: new_dof = new_dof[:, :, np.newaxis]
                target_dict[dof_key] = new_dof
            else:
                print(f"⚠️ Dimensions do not match {src_path}: Data {original_dof.shape[1]} vs Config {len(G1_MIRROR_PERM)}")

        # 2. Process Root Pos
        pos_key = None
        for k in ['root_trans_offset', 'root_pos', 'trans']:
            if k in target_dict: pos_key = k; break
        if pos_key:
            target_dict[pos_key] = mirror_position(target_dict[pos_key], MIRROR_AXIS_DIM)

        # 3. Process Root Rot
        rot_key = None
        for k in ['root_rot', 'root_orient', 'root_quat']:
            if k in target_dict: rot_key = k; break
        if rot_key:
            target_dict[rot_key] = mirror_quaternion(target_dict[rot_key], MIRROR_AXIS_DIM)

        # save final_data
        joblib.dump(final_data, dst_path)
        
    except Exception as e:
        print(f"❌ Error processing {src_path}: {e}")

if __name__ == "__main__":
    if not os.path.exists(SRC_DIR):
        print(f"❌ error: NO dic: {SRC_DIR}")
        exit()
        
    # get all pkl files
    all_pkl_files = []
    for root, dirs, files in os.walk(SRC_DIR):
        for f in files:
            if f.endswith('.pkl'):
                all_pkl_files.append(os.path.join(root, f))
    
    if not all_pkl_files:
        print("❌ No .pkl files found, please check path")
        exit()

    print(f"🚀 Found {len(all_pkl_files)}  PKL files, Start processing...")
    print(f"   Input path: {SRC_DIR}")
    print(f"   Output path: {OUT_DIR}")
    
    for src_p in tqdm(all_pkl_files):
        rel_path = os.path.relpath(src_p, SRC_DIR)
        
        dst_p_orig = os.path.join(OUT_DIR, rel_path)
        dst_p_mirror = os.path.join(OUT_DIR, os.path.dirname(rel_path), 'M' + os.path.basename(rel_path))
        
        os.makedirs(os.path.dirname(dst_p_orig), exist_ok=True)
        
        try:
            joblib.dump(joblib.load(src_p), dst_p_orig)
            
            process_single_pkl(src_p, dst_p_mirror)
        except Exception as e:
            print(f"File Error {src_p}: {e}")
            
    print("\n✅ Process done！")
    print("Tips: If any file was skipped, please check joints dimensions.")