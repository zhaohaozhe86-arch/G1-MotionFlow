import os
import random

data_root = "datasets/unitreeg1/new_joint_vecs"  
output_dir = "datasets/unitreeg1"                
split_ratio = 0.95

files = [f[:-4] for f in os.listdir(data_root) if f.endswith('.npy')]
random.shuffle(files)

split_idx = int(len(files) * split_ratio)
train_files = files[:split_idx]
test_files = files[split_idx:]

def write_list(path, file_list):
    with open(path, 'w') as f:
        for name in file_list:
            f.write(name + '\n')
    print(f"Saved {len(file_list)} files to {path}")

write_list(os.path.join(output_dir, "train.txt"), train_files)
write_list(os.path.join(output_dir, "test.txt"), test_files)