#!/bin/bash

# 如果你没有 train.env 文件，可以直接注释掉这一行，前提是你已经激活了 conda 环境
source train.env 

INPUT_DIR="g1_data_pkl_mirrored"
OUTPUT_DIR="g1_data_npz_mirrored"
ROBOT_CFG="holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
PIPELINE="['filename_as_motionkey']"
TARGET_FPS=20

python \
    holomotion/src/motion_retargeting/gmr_to_humanml3d.py \
    io.robot_config=${ROBOT_CFG} \
    io.src_dir=${INPUT_DIR} \
    io.out_root=${OUTPUT_DIR} \
    processing.target_fps=${TARGET_FPS} \
    preprocess.pipeline=${PIPELINE} \
    ray.num_workers=16