import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm

DATA_DIR = './unitreeg1/new_joint_vecs'
SAVE_DIR = './unitreeg1/meta' 
JOINTS_NUM = 30

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ error: no files {DATA_DIR}")
        return

    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    data_list = []

    print(f"📊 loading data ({len(file_list)} files)...")
    
    for file in tqdm(file_list):
        path = pjoin(DATA_DIR, file)
        try:
            data = np.load(path)
            
            # safety check
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"⚠️ skip (NaN/Inf): {file}")
                continue
                
            data_list.append(data)
        except Exception as e:
            print(f"❌ loading error {file}: {e}")

    if not data_list:
        print("❌ no valid data, quit")
        return

    all_data = np.concatenate(data_list, axis=0)
    N, D = all_data.shape
    
    # Root(4: 1 Yaw + 2 XZVel + 1 Y) + RIC((J-1)*3) + Rot(J*6) + LocalVel(J*3) + Contacts(2)
    expected_D = 4 + (JOINTS_NUM - 1) * 3 + JOINTS_NUM * 6 + JOINTS_NUM * 3 + 2
    
    print(f"✅ Loading data completed. Total frames: {N}, dimentioins: {D}")
    
    if D != expected_D:
        print(f"❌ Error: Wrong feature dimentions: Expecting {expected_D}, actual {D}。")
        return 

    Mean = all_data.mean(axis=0)
    Std = all_data.std(axis=0)

    idx1 = 1  # Root Yaw (1)
    idx2 = 3  # Root XZ vel (2)
    idx3 = 4  # Root Y height (1)
    idx4 = 4 + (JOINTS_NUM - 1) * 3               # RIC_data (J-1)*3
    idx5 = idx4 + JOINTS_NUM * 6                  # Rot_data (J*6) 
    idx6 = idx5 + JOINTS_NUM * 3                  # Local_vel (J*3)
    idx7 = idx6 + 2                               # Contacts (2)

    tiny_mask = Std < 1e-4
    if tiny_mask.any():
        print(f"🛡️ auto fix {tiny_mask.sum()} ")
        Std[tiny_mask] = 1.0

    Std[0:idx1] = Std[0:idx1].mean() / 1.0
    Std[idx1:idx2] = Std[idx1:idx2].mean() / 1.0
    Std[idx2:idx3] = Std[idx2:idx3].mean() / 1.0
    Std[idx3:idx4] = Std[idx3:idx4].mean() / 1.0  # RIC_data
    Std[idx4:idx5] = Std[idx4:idx5].mean() / 1.0  # Rot_data
    Std[idx5:idx6] = Std[idx5:idx6].mean() / 1.0  # Local_vel
    Std[idx6:idx7] = Std[idx6:idx7].mean() / 1.0  # Contacts

    # Verify whether the slice is completely covered
    assert idx7 == Std.shape[-1], f"Slice logic error: Expecting {Std.shape[-1]}, actual {idx7}！"

    # Save files
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(pjoin(SAVE_DIR, 'Mean.npy'), Mean)
    np.save(pjoin(SAVE_DIR, 'Std.npy'), Std)
    
    print(f"\n Preview Std ")
    print(f"Root Yaw:       {Std[0]:.4f}")
    print(f"Root XZ Vel:    {Std[1]:.4f}")
    print(f"RIC Data:   {Std[idx3:idx3+3]}")
    print(f"Rot Data:   {Std[idx4:idx4+3]}")
    print(f"\n✅ Mean.npy & Std.npy saved: {SAVE_DIR}")

if __name__ == "__main__":
    main()