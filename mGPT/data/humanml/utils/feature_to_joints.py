import torch
import numpy as np

def qinv(q):
    a = q.clone()
    a[..., 1:] *= -1
    return a

def qmul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def qrot(q, v):
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_inv_val = qinv(q)
    res = qmul(qmul(q, q_v), q_inv_val)
    return res[..., 1:]

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def matrix_to_quaternion(matrix):
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    scale = 0.25 / w
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) * scale
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) * scale
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) * scale
    return torch.stack([w, x, y, z], dim=-1)

def yaw_to_quaternion(yaw):
    half_yaw = yaw / 2.0
    w = torch.cos(half_yaw)
    y = torch.sin(half_yaw)
    x = torch.zeros_like(w)
    z = torch.zeros_like(w)
    return torch.stack([w, x, y, z], dim=-1)

def recover_g1_motion(features, mean=None, std=None):

    if mean is not None and std is not None:
        device = features.device
        t_mean = torch.tensor(mean, device=device, dtype=features.dtype)
        t_std = torch.tensor(std, device=device, dtype=features.dtype)
        features = features * t_std + t_mean

    B, T, D = features.shape
    
    assert (D - 3) % 12 == 0, f"{D} Error "
    J = (D - 3) // 12
    
    idx1 = 1
    idx2 = 3
    idx3 = 4
    idx4 = 4 + (J - 1) * 3
    idx5 = idx4 + J * 6
    
    yaw_vel = features[..., 0:idx1]            
    root_vel_xz = features[..., idx1:idx2]     
    root_y = features[..., idx2:idx3]          
    ric_data = features[..., idx3:idx4].view(B, T, J-1, 3)     
    rot_data = features[..., idx4:idx5].view(B, T, J, 6) 
    
    root_rots_yaw_only = torch.zeros((B, T, 4), device=features.device)
    root_pos = torch.zeros((B, T, 3), device=features.device)
    
    q_curr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=features.device).repeat(B, 1)
    pos_curr = torch.zeros((B, 3), device=features.device)
    
    for t in range(T):
        if t > 0:
            q_delta = yaw_to_quaternion(yaw_vel[:, t-1, 0])
            q_curr = qmul(q_curr, q_delta)
            q_curr = q_curr / q_curr.norm(dim=-1, keepdim=True)
            
            v_loc_xz = root_vel_xz[:, t-1]
            v_loc_3d = torch.stack([v_loc_xz[:, 0], torch.zeros_like(v_loc_xz[:, 0]), v_loc_xz[:, 1]], dim=-1)
            q_prev = root_rots_yaw_only[:, t-1]
            v_glob = qrot(q_prev, v_loc_3d)
            
            pos_curr[:, 0] += v_glob[:, 0]
            pos_curr[:, 2] += v_glob[:, 2]
            
        pos_curr[:, 1] = root_y[:, t, 0]
        
        root_rots_yaw_only[:, t] = q_curr.clone()
        root_pos[:, t] = pos_curr.clone()

    joint_rot_relative_mat = rotation_6d_to_matrix(rot_data)
    joint_rot_relative = matrix_to_quaternion(joint_rot_relative_mat)
    
    q_root_expanded_for_rot = root_rots_yaw_only.unsqueeze(2).repeat(1, 1, J, 1)
    global_rotations = qmul(q_root_expanded_for_rot, joint_rot_relative)

    global_positions = torch.zeros((B, T, J, 3), device=features.device)
    global_positions[:, :, 0, :] = root_pos
    
    q_root_expanded_for_pos = root_rots_yaw_only.unsqueeze(2).repeat(1, 1, J-1, 1)
    local_pos = qrot(q_root_expanded_for_pos, ric_data)
    global_positions[:, :, 1:, :] = root_pos.unsqueeze(2) + local_pos
    
    return global_positions, global_rotations

if __name__ == "__main__":
    B, T, D = 2, 10, 363 
    dummy_features = torch.randn(B, T, D)
    
    positions, rotations = recover_g1_motion(dummy_features)
    print(f"✅")
    print(f"Global position Shape: {positions.shape}  (B, T, J, 3)")
    print(f"Global rotation Shape: {rotations.shape}  (B, T, J, 4)")