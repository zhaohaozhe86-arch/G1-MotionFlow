import numpy as np
import os
from tqdm import tqdm
from os.path import join as pjoin

INPUT_DIR = './joints_with_rot'
OUTPUT_DIR = './unitreeg1/new_joint_vecs'
L_FOOT_IDX = 6
R_FOOT_IDX = 12

def qinv(q):
    a = q.copy()
    a[..., 1:] *= -1
    return a

def qmul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)

def qrot(q, v):
    q_v = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    q_inv = qinv(q)
    res = qmul(qmul(q, q_v), q_inv)
    return res[..., 1:]

def quat_to_6d(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y**2 + z**2)
    r10 = 2 * (x*y + w*z)
    r20 = 2 * (x*z - w*y)
    r01 = 2 * (x*y - w*z)
    r11 = 1 - 2 * (x**2 + z**2)
    r21 = 2 * (y*z + w*x)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)

def get_yaw_from_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2 * (w*y + x*z), 1 - 2 * (y**2 + z**2))
    return yaw

def yaw_to_quaternion_np(yaw):
    half_yaw = yaw / 2.0
    w = np.cos(half_yaw)
    y = np.sin(half_yaw)
    x = np.zeros_like(w)
    z = np.zeros_like(w)
    return np.stack([w, x, y, z], axis=-1)

def process_file(positions, rotations):
    T, J, _ = positions.shape
    
    root_pos = positions[:, 0:1, :] 
    root_rot_raw = rotations[:, 0:1, :] 
    
    root_yaw = get_yaw_from_quat(root_rot_raw) 
    root_rot_yaw_only = yaw_to_quaternion_np(root_yaw) 

    # [Root Height] (1 dim)
    root_y = root_pos[:-1, 0, 1:2] 

    # [Root Linear Velocity XZ] (2 dims)
    global_vel = root_pos[1:] - root_pos[:-1] 
    local_vel = qrot(qinv(root_rot_yaw_only[:-1]), global_vel).squeeze(1)
    root_linear_vel_xz = local_vel[:, [0, 2]] 

    # [Root Angular Velocity] (1 dim)
    root_delta_rot = qmul(qinv(root_rot_yaw_only[:-1]), root_rot_yaw_only[1:]) 
    r_velocity = get_yaw_from_quat(root_delta_rot).reshape(-1, 1) 

    # RIC position feature (J-1 * 3 dims = 87 dims)
    local_pos = positions - root_pos 
    local_pos_invariant = qrot(qinv(root_rot_yaw_only), local_pos) 
    ric_data = local_pos_invariant[:-1, 1:, :].reshape(T-1, -1) 

    # Relative joints rotation feature (J * 6 dims = 180 dims)
    joints_rot = rotations[:-1, :, :] 
    joints_relative = qmul(qinv(root_rot_yaw_only[:-1]), joints_rot) 
    rot_data = quat_to_6d(joints_relative).reshape(T-1, -1) 

    # Local velocity feature(J * 3 dims = 90 dims)
    global_joint_vel = positions[1:] - positions[:-1]
    local_joint_vel = qrot(qinv(root_rot_yaw_only[:-1]), global_joint_vel)
    local_vel_data = local_joint_vel.reshape(T-1, -1)

    # Feet contect (2 dims)
    l_foot_vel = np.sum((positions[1:, L_FOOT_IDX] - positions[:-1, L_FOOT_IDX])**2, axis=-1)
    r_foot_vel = np.sum((positions[1:, R_FOOT_IDX] - positions[:-1, R_FOOT_IDX])**2, axis=-1)
    contacts = (np.stack([l_foot_vel, r_foot_vel], axis=-1) < 0.002).astype(np.float32)

    # 1 + 2 + 1 + 87 + 180 + 90 + 2 = 363
    data = np.concatenate([
        r_velocity,          # 1
        root_linear_vel_xz,  # 2
        root_y,              # 1
        ric_data,            # (J-1)*3 
        rot_data,            # J*6 
        local_vel_data,      # J*3 
        contacts             # 2
    ], axis=-1) 
    
    return data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.npy') and not f.endswith('_rot.npy')]
    
    print(f"🚀 Feature Extraction (363 Dims), Output: {OUTPUT_DIR}")
    
    for file_name in tqdm(files):
        try:
            pos_path = pjoin(INPUT_DIR, file_name)
            rot_path = pjoin(INPUT_DIR, file_name.replace('.npy', '_rot.npy'))
            if not os.path.exists(rot_path): continue

            positions = np.load(pos_path)   
            rotations = np.load(rot_path)   
            rotations = rotations[..., [3, 0, 1, 2]] 

            features = process_file(positions, rotations)
            np.save(pjoin(OUTPUT_DIR, file_name), features)
            
        except Exception as e:
            print(f"❌ Error {file_name}: {e}")

if __name__ == "__main__":
    main()