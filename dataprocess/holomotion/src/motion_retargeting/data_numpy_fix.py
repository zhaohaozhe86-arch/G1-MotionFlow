import sys
import os
import glob
import pickle
import numpy
from tqdm import tqdm

# ==========================================
# 核心补丁：让当前环境能读懂 NumPy 2.0 的数据
# ==========================================
print("Applying NumPy compatibility patch...")
try:
    # 强制将 numpy._core 映射回 numpy.core
    sys.modules['numpy._core'] = numpy.core
    # 处理 multiarray
    if hasattr(numpy.core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
    print("Patch applied successfully.")
except Exception as e:
    print(f"Patch warning: {e}")

# ==========================================
# 数据转换逻辑
# ==========================================
def fix_all_files(src_dir):
    # 搜索所有的 pkl 文件（包括子文件夹）
    files = glob.glob(os.path.join(src_dir, "**/*.pkl"), recursive=True)
    
    if not files:
        # 如果不支持递归（旧版 python），尝试只搜根目录
        files = glob.glob(os.path.join(src_dir, "*.pkl"))
    
    print(f"Found {len(files)} files in '{src_dir}'. Starting conversion...")
    
    success_count = 0
    fail_count = 0

    for fpath in tqdm(files):
        try:
            # 1. 用补丁环境读取（Load）
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            
            # 2. 用当前环境写回（Dump）
            # 这样文件就变成了标准的 NumPy 1.x 格式，以后谁都能读
            with open(fpath, 'wb') as f:
                pickle.dump(data, f)
                
            success_count += 1
        except Exception as e:
            print(f"\nFailed to process {fpath}: {e}")
            fail_count += 1

    print("\n" + "="*30)
    print(f"Processing Complete!")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print("="*30)

if __name__ == "__main__":
    # 指定你的数据文件夹名字
    INPUT_DIR = "g1_data_pkl" 
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' not found!")
    else:
        fix_all_files(INPUT_DIR)