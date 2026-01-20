# human-pose-manipulation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b1nXrFfXnuIDzx3F7nxrcH4CUK7ycDGC?usp=sharing)

⚠️ Note:
If the notebook cannot be rendered directly on GitHub,
please open it using Google Colab via the badge above.

# Adaptive Text-Driven Human Pose Manipulation  
## LoRA-Augmented InstructPix2Pix Pipelines

This repository implements a **resource-efficient text-guided human pose manipulation system** based on **Stable Diffusion v1.5** and **InstructPix2Pix**, enhanced with **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning.

The primary goal of this project is to **edit existing images**—specifically transforming human poses (e.g., *standing → sitting*)—while **preserving subject identity, clothing details, and background consistency**, even under **limited computational resources** such as Google Colab GPUs.

---

## ✨ Key Features

- **Text-Driven Pose Editing**  
  Edit human poses using natural language instructions (e.g., *"make the person sit down"*) without explicit pose maps or manual masks.

- **Parameter-Efficient Fine-Tuning (LoRA)**  
  Adapts InstructPix2Pix by fine-tuning less than **1% of total parameters**, enabling training on consumer-grade GPUs.

- **Identity & Background Preservation**  
  Reduces semantic drift by freezing base model weights and learning only pose-specific transformations.

- **Optimized Inference Pipeline**
  - FP16 mixed precision
  - Memory-efficient attention via `xformers`
  - Fast sampling using `DPMSolverMultistepScheduler`  
  → ~2–5 seconds per image

- **Context-Aware Synthetic Dataset**  
  Procedurally generated paired images with controlled backgrounds to isolate pose-related geometric changes.

---

## 🧠 Model Architecture

- **Base Model**: Stable Diffusion v1.5  
- **Editing Framework**: InstructPix2Pix  
- **Fine-Tuning Method**: Low-Rank Adaptation (LoRA, rank = 8)  
- **Latent Representation**: Variational Autoencoder (VAE)  
- **Text Encoder**: CLIP  
- **Scheduler**: DPMSolverMultistepScheduler  

---

## 📊 Evaluation Metrics

The system is evaluated using both quantitative and qualitative metrics:

- **SSIM** – Structural consistency and identity preservation  
- **PSNR** – Pixel-level similarity  
- **LPIPS** – Perceptual similarity  
- **CLIP Score / Directional CLIP** – Text-image alignment  

Experimental results demonstrate strong structural consistency (SSIM ≈ 0.77) despite significant geometric pose transformations.

---

## 🛠️ Environment & Requirements

- Python 3.9+
- PyTorch
- `diffusers`
- `transformers`
- `peft`
- `bitsandbytes`
- `xformers`

**Recommended Hardware:**  
Google Colab (Tesla T4 / A100)

Refer to the notebook for installation and execution details.

---

## 📁 Repository Structure

