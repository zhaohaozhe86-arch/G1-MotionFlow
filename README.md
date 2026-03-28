> 🌐 **English** | [简体中文](README_zh-CN.md)

---

<div align="center">
    <h1> G1-Motionflow: Natural Language-Driven Humanoid Robot Motion Generation Framework </h1>
</div>

<div align="center">
    <h2> Bridging the Gap Between High-Level Human Intent and Low-Level Physical Execution </h2>
</div>

<div align="center">
    <p align="center">
      <a href="#-introduction">Introduction</a> •
      <a href="#-quick-start">Quick Start</a> •
      <a href="#-technical-architecture">Technical Architecture</a> •
      <a href="#-key-breakthrough">Key Breakthrough</a> •
      <a href="#-results">Results</a> •
      <a href="#-future-work">Future Work</a> •
      <a href="#-license">License</a>
    </p>
</div>

https://github.com/user-attachments/assets/982983d2-fc45-4eb5-b54d-033501930c35

## 🏃 Introduction

Current mainstream general controllers primarily rely on predefined trajectories or continuous teleoperation, lacking runtime flexibility and robot autonomy. The reference trajectories inferred by high-level motion generation models often have a distribution shift from the high-quality motion capture data used to train low-level controllers. Direct deployment can cause the robot to lose balance or fall in the real physical world.

**G1-Motionflow** explores real-time interactive motion generation and control based on natural language, bridging the gap between "high-level human intent expression" and "low-level robot physical execution." This allows the robot to break free from fixed command constraints, enabling free intent expression in single tasks and smooth transitions across multiple tasks.

<div align="center">
    <img width="100%" alt="System Architecture Diagram" src="assets/G1-MotionFlow_Architechture.png">
</div>

## ⚡ Quick Start

<details>
  <summary><b>Setup and download</b></summary>

### 1. Environment Setup

First, configure the Python virtual environment and install the necessary dependencies and large-file download tools:

```bash
conda create python=3.10 --name g1-flow
conda activate g1-flow
pip install -r requirements.txt

# Install gdown to bypass Google Drive large file download limits
pip install gdown 
````

### 2. Download Pretrained Models

Run the following scripts to download the required pretrained language and speech models from HuggingFace. The models will be automatically placed in the `deps/` directory:

```bash
bash prepare_t5.sh
bash prepare_whisper.sh
```

### 3. Download Project Assets

Run the following scripts to retrieve the pretrained motion generation weights, tracker configurations, and raw data from Google Drive. The `.tar.xz` archives will be automatically downloaded, extracted to the main directory, and cleaned up:

```bash
# Get experimental configurations and pretrained weights (generates experiments/ folder)
bash download_experiments.sh  

# Get low-level tracker model and configurations (generates tracker/ folder)
bash download_tracker.sh      

# [Optional] Get raw motion sequences retargeted to G1 (downloads raw_data.tar.xz only)
bash download_raw_data.sh     
```

</details>

## ⚙️ Technical Architecture

Addressing the complete pipeline from natural language to physical robot deployment, this framework consists of two core modules: physics-oriented dataset construction and a three-stage end-to-end model training framework.

### 1. Physics-Oriented Dataset Construction

<details>
<summary><b>Construction Process</b></summary>

Current large-scale text-motion datasets are mostly based on the SMPL skeleton, lacking real-world physical constraints and making them unsuitable for direct robot imitation.

  * **Cross-Morphology Retargeting (SMPL to Robot):** Based on the AMASS dataset. The SMPL joint rotation angles are converted into spatial positions via forward kinematics, and then retargeted to the 29 joints of the Unitree G1 robot using the GMR tool. The dataset is built in conjunction with the text annotations from HumanML3D.
    <div align="center">
    <img width="100%" alt="Retargeting Demonstration" src="assets/Retarget.png">
    </div>
  * **Mirror Augmentation and Kinematics:** Developed automated detection and alignment scripts to unify the global poses. After mirroring the joint spatial positions, inverse kinematics, physical flexible interpolation, and downsampling are applied to generate 363-dimensional data containing joint positions, angles, and foot contact information.
  * **Physical Simulation Filtering:** The full 24,788 generated sequences were fed into a pre-trained tracking model and tested in MuJoCo. Approximately 20% of the low-quality data that caused the robot to fall was eliminated.

</details>

### 2. Three-Stage End-to-End Training Framework

<details>
<summary><b>Training Process</b></summary>
<div align="center">
<img width="100%" alt="Model Architecture" src="assets/Model.png">
</div>

  * **Stage 1: Motion Feature Discretization (VQ-VAE).** The filtered 363-dimensional continuous motion sequences are fed into a VQ-VAE for feature extraction and reconstruction, compressing and mapping them into a Codebook containing 512 discrete Tokens.
  * **Stage 2: Semantic-to-Motion Generation (T5-Base).** The natural language text and motion Tokens are concatenated and input into the T5-base model. It learns the mapping rules from semantics to discrete motion commands in an autoregressive manner.
  * **Stage 3: Low-Level Control and Physical Deployment (RL Tracking).** Referencing the `rl_sar` open-source framework, a whole-body motion tracking policy is trained using the PPO algorithm in a physical simulation environment. By setting composite reward functions such as joint position tracking error, velocity penalties, and torque limits, it achieves robust deployment to real hardware. Trajectories generated by the generator are heavily introduced during the training phase to proactively adapt to noise, enhancing the system's closed-loop robustness.
    <div align="center">
    <img width="100%" alt="Data Processing for Distribution Shift" src="assets/dataprocess.png">
    </div>

</details>

## 🔬 Key Breakthrough: Resolving the "Zero-Drift" Issue

In the early stages of VQ-VAE training (when only joint rotation angles were retained), motion reconstruction suffered from a "zero-drift" phenomenon: cumulative errors generated over time led to posture distortion.

<div align="center">
<img src="assets/wavehand1.gif" width="24%" alt="zero-drift">
<img src="assets/wavehand2.gif" width="24%" alt="no zero-drift">
</div>

<details>
<summary><b>Solution</b></summary>

  * **Solution:** Drawing on multi-sensor fusion concepts, an **"Active Over-Redundancy"** strategy was adopted.
  * **Implementation:** Both joint rotation angles and joint spatial position information were introduced into the feature vector simultaneously, allowing them to constrain each other within the loss function.
  * **Loss Function:** $Loss = L_{recon} + L_{vel} + L_{commit}$ (combining reconstruction loss, velocity loss, and commitment loss). This effectively corrected the cumulative deviation on the kinematic chain and completely eliminated zero-drift.

</details>

## 📊 Results & Demos

https://github.com/user-attachments/assets/2f44abc3-ef92-49f7-a94a-68f3126e831b

The complete closed-loop from "command-generation-actuation" has been achieved:

1.  **Ultra-Low Latency and High Efficiency:** The framework has been lightened and supports one-click execution on a laptop. The end-to-end time from natural language input to Token generation and driving the G1 robot takes **less than 1 second**.
2.  **Robust Semantic Understanding:** The T5-base model enables the understanding of highly abstract commands (e.g., "walking on a very slippery road, treading on thin ice", "poised like a fully drawn bow"). The generated motion sequences break the limitations of traditional predefined motion libraries, balancing diversity and physical stability.

## 🚀 Limitations and Future Work

The current model is limited by short-horizon motion generation and lacks instruction degrees of freedom.

The next phase plans to introduce a motion synthesis architecture based on part-level decomposition:

  * **Spatial Dimension (Part-Level Representation):** Decouple the robot into independent parts such as the left arm, right arm, torso, and legs. Use LLMs to rewrite "walking" into detailed part descriptions (e.g., "legs move forward alternately, arms swing accordingly"), achieving multi-task parallel control through independent generation and spatial coupling of parts.
  * **Temporal Dimension (Long-Horizon Task Decomposition):** Introduce LLMs as "high-level planners" to break down long-horizon task instructions into short-horizon part-level motion primitives. Break the dataset frame limits through a "building block" approach to construct long sequences.

## 📖 Acknowledgments

This project incorporates code and assets from the following upstream projects:

  * [MotionGPT](https://github.com/OpenMotionLab/MotionGPT.git) (Copyright (c) 2023 OpenMotionLab)
  * [TextOp](https://github.com/TeleHuman/TextOp.git) (Copyright (c) 2025 TextOp Team)

## ⚖️ License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).