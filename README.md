# Segment Anything Model 2 (SAM2) - Video Segmentation Guide

This repository provides a setup guide and script for running **SAM2** to perform video segmentation using user-provided clicks.

---

## 1. Cloning the Repository
Start by cloning the repository:

```bash
git clone https://github.com/YOUR_USERNAME/notion-sam2-setup.git
cd notion-sam2-setup
```

---

## 2. Setup Steps

### Install Dependencies
Ensure you have **Python 3.10** installed and create a virtual environment:

```bash
conda create -n sam2 python=3.10 -y
conda activate sam2
```

### Install Required Packages
```bash
pip install torch==2.3.1 torchvision==0.18.1
```

### Clone the SAM2 Repository and Install Dependencies
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
pip install -e ".[demo]"
```

### Install NVIDIA CUDA Toolkit (if using a GPU)
```bash
sudo apt install -y nvidia-cuda-toolkit
```

---

## 3. Download Input Video Frames
Since the dataset is too large for GitHub, **download it from Google Drive**:

```bash
pip install gdown
gdown --fuzzy 'https://drive.google.com/file/d/1REWEAeMP8W1Y0NqU9M1pBQYSdDXZvzYH/view?usp=sharing' -O /home/earthsense/Documents/collection-130624_040630_zed_camera_vis.zip
unzip /home/earthsense/Documents/collection-130624_040630_zed_camera_vis.zip -d /home/earthsense/Documents
mv /home/earthsense/Documents/collection-130624_040630_zed_camera_vis_short /home/earthsense/Documents/
```

---

## 4. Download Model Checkpoints
```bash
cd segment-anything-2/checkpoints
./download_ckpts.sh
cd ..
```

---

## 5. Running the Script
Since `video_demo_script.py` is already in this repo, you can directly run:

```bash
python video_demo_script.py
```

Upon running, you'll be prompted to start segmenting the first frame of the video. Follow the instructions:

1. **Left Click** → Positive Click (green star)
2. **Right Click** → Negative Click (red star)
3. **Middle Click** → Start a new object ID

Once you’ve segmented all objects, exit the interface by clicking the **"X"** in the top-right corner.

---

## 6. Outputs
- Segmented video frames will be saved in:
  ```
  /home/earthsense/segment-anything-2/segmented_video_frames/
  ```
- The final segmented video will be saved at:
  ```
  /home/earthsense/segment-anything-2/output_video.mp4
  ```
- The segmentation mask information will be stored in:
  ```
  /home/earthsense/segment-anything-2/segmentation_info.json
  ```