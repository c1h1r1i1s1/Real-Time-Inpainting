# Video Inpainting Model (DSTT Variant with Memory)

This repository contains the implementation of a **real-time transformer-based video inpainting model** designed for privacy-preserving **diminished reality** applications.  
The project includes:
- A **PyTorch model definition** (with memory-augmented transformer blocks).  
- **Training scripts** with a staged curriculum.  
- **Data scraping scripts** for obtaining synthetic video training data (e.g., GTA-V).  
- Export utilities for **TorchScript** and **ONNX**, enabling deployment as a **TensorRT engine** in production pipelines.
- DSTT_OM_RT engine file for immediate use with inpainting_manager backend server. *Note: This engine has not been fully trained due to cost and time restrictions. Please feel free to re-train/continue training for more robust and accurate inpainting results.*

---

## 🚀 Overview

This model is a **Decoupled Spatial–Temporal Transformer (DSTT) variant** with custom modifications for efficiency and memory handling:

- **Hierarchical Encoder** with residual channel grouping.  
- **Spatial Transformer Blocks with Memory**: Attention across spatial tokens and past memory slots.  
- **Temporal Transformer Blocks with Memory**: Temporal attention with grouped spatial tokens, designed for streaming sequences.  
- **Vec2PatchAlternative** module for token-to-patch reconstruction with channel reduction.  
- **ONNX/TensorRT Export Support** for high-performance inference.

The model operates on **360×640 video frames**, maintaining a low-latency, GPU-efficient pipeline for continuous frame-by-frame inpainting.

---

## 🧩 Model Structure & Modifications

### 1. Encoder
- **Conv layers** reduce input resolution (360×640 → 90×160).  
- **HierarchyEncoder** introduces grouped residual connections for efficient feature reuse.  
- **Patch2Vec**: Projects encoder features into a token sequence of length `1620` (30×54 grid).  

### 2. Transformer Blocks
- **Alternating design**:
  - **SpatialTransformerBlockWithMemory**: Attends over current frame tokens + past memory.  
  - **TemporalTransformerBlockWithMemory**: Attends across time by grouping spatial patches.  
- **Memory mechanism**: Stores past frame tokens, dropping the oldest and appending the newest at each step.  

### 3. Decoder
- **Vec2PatchAlternative** reconstructs image patches from transformer tokens.  
- Features fused back into encoder output.  
- **Deconv layers** upsample back to 360×640.  
- Final **tanh activation** ensures normalized output.  

### 4. Modifications vs Standard DSTT
- Explicit **center cropping** to enforce consistent feature map sizes.  
- Memory passed per transformer block (`num_blocks × memory_slots × seq_len × hidden`).  
- Custom **Vec2PatchAlternative** layer with 1×1 channel reduction for efficiency.  
- Exportable via **TorchScript + ONNX** with dynamic axes for batch size and sequence length.  

---

## 🏋️ Training Curriculum

Training proceeds in **stages**, gradually increasing the diversity and difficulty of the data and masks.  
The script implements both **curriculum datasets** and **mask generation strategies** to stabilize learning.  

### 1. Data Curriculum
- **Stage 1**  
  - Datasets:  
    - **GTA-IM Processed** (synthetic gameplay footage).  
    - **MIT Indoor Scenes Processed** (real-world stills).  
  - Goal: Teach the model **spatial inpainting** and short temporal consistency.  
  - Epochs: 20.  
  - Loader: Combines datasets via `ConcatDataset`.  

- **Stage 2**  
  - Dataset: **SUN3D_frames** (RGB-D video sequences).  
  - Goal: Learn **multi-frame temporal consistency** with real-world camera motion.  
  - Suggested Epochs: 5.  

- **Stage 3**  
  - Dataset: **Thumos14 Processed** (sports/action videos).  
  - Goal: Handle **fast motion, dynamic objects, and varied environments**.  
  - Suggested Epochs: 5.  

Each stage builds upon the previous, with checkpoints saved at the end of training.  

### 2. Mask Generation Curriculum
The training loop applies **different types of masks** to simulate holes:  

- **Rectangular masks**: Cover 5–20% of the image.  
- **Free-form masks**: Generated with random strokes (irregular shapes).  
- **Temporal perturbations**:  
  - With 50% chance, the same mask is applied across all frames.  
  - Otherwise, the mask is slightly shifted per frame (`perturb_mask`), simulating moving occlusions.  

This encourages the model to handle both **stable occlusions** (e.g., stationary objects) and **dynamic occlusions** (e.g., moving people).  

### 3. Loss Function
- **Weighted L1 Loss**:
  - Hole regions (`mask == 1`) weighted separately from valid regions (`mask == 0`).  
  - Balances reconstruction of missing areas with preservation of unmasked regions.  

### 4. Memory in Training
- Each sequence processes **5 frames**.  
- A **shared memory tensor** (`num_blocks=8, memory_slots=2`) carries token history across frames.  
- Memory is **detached** at each timestep to prevent unrolled backpropagation across time (stabilizes training).  

### 5. Training Setup
- **Optimizer**: Adam (`lr=1e-4`, betas=(0.0, 0.99)).  
- **Batch size**: 2 (fits within 16–24 GB VRAM).  
- **Mixed precision (AMP)**: Enabled with `torch.autocast` + gradient scaling for faster training.  
- **Device**: CUDA recommended (tested on AWS A10G/A100).

### 6. Example Training Flow
```bash
# Stage 1
python train.py
# Produces checkpoints/stage1_final.pth

# Stage 2
# Uncomment SUN3D loader + training loop in train.py
# Produces checkpoints/stage2_final.pth
```

---

## ⚙️ System Requirements

### Training
- **GPU**:  
  - Minimum: 16 GB VRAM (NVIDIA V100, RTX A4000).  
  - Recommended: 24–32 GB VRAM (A10G, A100, RTX 3090/4090).  
- **Frameworks**:  
  - PyTorch ≥ 1.13 with CUDA 11.6+.
- **Time**:  
  - Stage 1: ~1–2 days on A100.  
  - Stage 2: ~3–5 days depending on sequence length. 

### Inference
- **GPU**: RTX 3070 or better for real-time inference.  
- **Memory slots**: Tunable (default: 2).  
- **Optimized deployment** requires TensorRT engine.  

---

## 🔄 Export & Deployment

### 1. Export to ONNX
The main script already includes an ONNX export step:

```bash
python inpaint_model.py
```
Produces:
```bash
DSTT_OM_RT.onnx
```

### 2. Convert ONNX → TensorRT
Use trtexec (bundled with TensorRT):
```bash
trtexec --onnx=DSTT_OM_RT.onnx \
        --saveEngine=inpaint_fp16.engine \
        --fp16 \
        --workspace=4096
```

### 3. Run in Application
Load TensorRT engine inside your C++/Python pipeline.
This integrates with the inpaint_manager repo for live mixed-reality inpainting.

---

🏆 Achievements
- Achieves real-time inference ~30fps at 360×640 resolution on RTX 3070+.
- Robust performance on synthetic (GTA-V) and real-world videos.
- Integrated seamlessly with Unity/Meta Quest pipelines for MR privacy applications.
