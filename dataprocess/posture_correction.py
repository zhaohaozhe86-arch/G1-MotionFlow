import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

INDEX_PATH = './index_augmented.csv'
GMR_OUTPUT_DIR = './g1_data_npz_mirrored/clips'
FINAL_OUTPUT_DIR = './joints_with_rot_v2' 
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

IDX_PELVIS = 0
IDX_TORSO = 15
IDX_L_SHOULDER = 17 
IDX_R_SHOULDER = 23
IDX_L_HIP = 1       # left_hip_pitch_link
IDX_R_HIP = 7       # right_hip_pitch_link

X_IS_LEFT = True 

DEBUG_TARGET = "000009.npy" 

def get_gmr_filename(source_path):
    clean_path = source_path.replace('./pose_data_g1/', '').replace('.npy', '').replace('.pkl', '')
    return clean_path.replace('/', '_') + '.npz'

def process_motion_with_rot(pos, rot, debug=False):
    pos = pos[..., [1, 2, 0]]
    rot = rot[..., [1, 2, 0, 3]] # Quaternion x,y,z mapping
    
    if debug: print("[Step 1] Coordinate axis mapping completed!")

    # Ensure Upright
    pose = pos[0]
    y_diff = pose[IDX_TORSO, 1] - pose[IDX_PELVIS, 1]
    
    if y_diff < 0:
        if debug: print(f"[Step 2] Detected inverted -> rotate 180 degrees correction.")
        r_fix = R.from_euler('z', 180, degrees=True)
        
        flat_pos = pos.reshape(-1, 3)
        pos = r_fix.apply(flat_pos).reshape(pos.shape)
        
        flat_rot = rot.reshape(-1, 4)
        r_old = R.from_quat(flat_rot)
        r_new = r_fix * r_old 
        rot = r_new.as_quat().reshape(rot.shape)
    
    # Robust Yaw Alignment
    pose0 = pos[0]
    
    # Left - Right
    vec_shoulder = pose0[IDX_L_SHOULDER] - pose0[IDX_R_SHOULDER]
    vec_hip = pose0[IDX_L_HIP] - pose0[IDX_R_HIP]
    vec_combined = vec_shoulder + vec_hip
    
    vec_flat = np.array([vec_combined[0], 0, vec_combined[2]]) 
    norm = np.linalg.norm(vec_flat)
    if norm < 1e-6:
        vec_flat = np.array([vec_shoulder[0], 0, vec_shoulder[2]])
    
    vec_flat /= (np.linalg.norm(vec_flat) + 1e-8)
    
    # Calculate the current relative x-axis angle
    angle = np.arctan2(vec_flat[2], vec_flat[0])
    rot_angle = -angle 
    
    if abs(rot_angle) > 1e-4:
        if debug: print(f"[Step 3] foundation rotation {np.degrees(rot_angle):.2f}°")
        r_align = R.from_euler('y', rot_angle, degrees=False)
        
        flat_pos = pos.reshape(-1, 3)
        pos = r_align.apply(flat_pos).reshape(pos.shape)
        
        flat_rot = rot.reshape(-1, 4)
        r_old = R.from_quat(flat_rot)
        r_new = r_align * r_old 
        rot = r_new.as_quat().reshape(rot.shape)

    # Force Face Z+
    pose_final_0 = pos[0]
    l_x = pose_final_0[IDX_L_HIP, 0]
    r_x = pose_final_0[IDX_R_HIP, 0]
    
    # Judgement: If X=Left, the X of Left_Hip has to be > than the X of Right_Hip
    is_backward = (l_x < r_x) if X_IS_LEFT else (l_x > r_x)
    
    if is_backward:
        if debug: print(f"[Step 4] Detected direction reversed (Face -Z) -> Forcing 180 degrees rotation")
        r_180 = R.from_euler('y', 180, degrees=True)
        
        flat_pos = pos.reshape(-1, 3)
        pos = r_180.apply(flat_pos).reshape(pos.shape)
        
        flat_rot = rot.reshape(-1, 4)
        r_old = R.from_quat(flat_rot)
        r_new = r_180 * r_old
        rot = r_new.as_quat().reshape(rot.shape)

    # Zero Root XZ at Frame 0
    # Get the position of Pelvis in frame 0
    root_pos_0 = pos[0, IDX_PELVIS]
    
    # Construct offset: X and Z take the current value, Y take 0 (no change of height)
    offset = np.array([root_pos_0[0], 0, root_pos_0[2]])
    
    if debug:
        print(f"[Step 5] Zero the root: offset X={-offset[0]:.4f}, Z={-offset[2]:.4f}")
        print(f"         (original position: {root_pos_0})")

    # All frames, all joints minus the offset
    pos -= offset
    
    # validation
    if debug:
        new_root = pos[0, IDX_PELVIS]
        print(f"         (new position: {new_root}) -> should be [0, Y, 0]")
    
    flat_rot = rot.reshape(-1, 4)
    r_all = R.from_quat(flat_rot)
    
    r_0 = r_all[0]

    yaw_0 = r_0.as_euler('yxz')[0]
    
    if abs(yaw_0) > 1e-5:
        if debug:
            print(f"[Step 6] Residual Yaw correction: {-np.degrees(yaw_0):.4f}°")
            
        # construct correction matrix: turn -yaw_0 around the Y axis
        r_fix_yaw = R.from_euler('y', -yaw_0)
        
        # correct position (turn around the origin)
        flat_pos = pos.reshape(-1, 3)
        pos = r_fix_yaw.apply(flat_pos).reshape(pos.shape)
        
        # correct rotation
        # New_Rot = R_fix * Old_Rot
        r_new = r_fix_yaw * r_all
        rot = r_new.as_quat().reshape(rot.shape)

    return pos, rot

def main():
    if not os.path.exists(INDEX_PATH): 
        print(f"Error: Index path {INDEX_PATH} not found.")
        return
        
    df = pd.read_csv(INDEX_PATH)
    print(f"🚀 Start processing: Forcing the positive direction and zero the xy coordinates of the node...")
    
    success = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        new_name = row['new_name']
        is_target = (new_name == DEBUG_TARGET)
        
        try:
            gmr_name = get_gmr_filename(row['source_path'])
            gmr_path = os.path.join(GMR_OUTPUT_DIR, gmr_name)
            if not os.path.exists(gmr_path): continue
            
            data_npz = np.load(gmr_path)
            if 'pos' not in data_npz: continue
            
            motion_pos = data_npz['pos']
            motion_rot = data_npz['rot'] if 'rot' in data_npz else None
            
            # slice
            start_f, end_f = int(row['start_frame']), int(row['end_frame'])
            if end_f != -1 and end_f > start_f:
                motion_pos = motion_pos[start_f:end_f]
                if motion_rot is not None: motion_rot = motion_rot[start_f:end_f]
            else:
                motion_pos = motion_pos[start_f:]
                if motion_rot is not None: motion_rot = motion_rot[start_f:]
            
            if len(motion_pos) < 5: continue

            if is_target: print(f"\n=== DEBUG {new_name} ===")

            # Construct Dummy Rot
            if motion_rot is None:
                if is_target: print("Info: No rot found, using dummy.")
                motion_rot = np.zeros((motion_pos.shape[0], motion_pos.shape[1], 4))
                motion_rot[..., 3] = 1.0
                save_rot = False
            else:
                save_rot = True

            # Core Processing
            final_pos, final_rot = process_motion_with_rot(motion_pos, motion_rot, debug=is_target)

            # Save
            np.save(os.path.join(FINAL_OUTPUT_DIR, new_name), final_pos)
            
            if save_rot:
                rot_name = new_name.replace('.npy', '_rot.npy')
                np.save(os.path.join(FINAL_OUTPUT_DIR, rot_name), final_rot)
            
            if is_target: print("=== END ===\n")
            success += 1
                
        except Exception as e:
            if is_target: print(f"Error processing {new_name}: {e}")
            pass

    print(f"Done. Processed {success} clips.")

if __name__ == "__main__":
    main()