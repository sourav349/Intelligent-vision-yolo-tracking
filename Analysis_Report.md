# Object Detection & Vehicle Tracking — Analysis Report

**Project**: Minor Project 2 — YOLO11 Based Object Detection & Tracking  
**Dataset**: BDD100K (2k Subset)  
**Model**: YOLO11n (Nano)  
**Date**: April 2026  

---

## 1. Project Overview

This project implements a real-time **object detection and vehicle tracking** pipeline using the YOLO11 architecture. The system is trained on a curated subset of the BDD100K driving dataset and performs:

- Multi-class object detection (6 classes)
- Vehicle tracking across video frames using ByteTrack
- Directional vehicle counting (Up / Down) across virtual lines
- Speed estimation (pixel displacement per frame)
- Training metrics visualization

---

## 2. Dataset Description

### Source Dataset: BDD100K (10k Subset)
| Split | Original Count |
|-------|---------------|
| Train | 7,000 images  |
| Val   | 1,000 images  |
| Test  | 1,995 images  |
| **Total** | **~10,000 images** |

### Reduced Dataset: 2k Subset *(used for training)*
Randomly sampled (seed=42) from the 10k dataset for faster training on CPU/low-RAM machines.

| Split | Images | Labels |
|-------|--------|--------|
| Train | 1,400  | 1,400  |
| Val   |   400  |   400  |
| Test  |   200  |   200  |
| **Total** | **2,000** | **2,000** |

**Dataset Location:**
```
C:\Users\Shivani Agarwal\Downloads\Garv_minor2\Object\2k\
    ├── images\
    │   ├── train\   →  1400 images
    │   ├── val\     →   400 images
    │   └── test\    →   200 images
    └── labels\
        ├── train\   →  1400 .txt label files
        ├── val\     →   400 .txt label files
        └── test\    →   200 .txt label files
```

### Class Labels
| ID | Class Name  |
|----|-------------|
| 0  | car         |
| 1  | bus         |
| 2  | truck       |
| 3  | motorcycle  |
| 4  | bicycle     |
| 5  | person      |

### Data Format: YOLO Format
Each `.txt` label file contains one annotation per line:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized between 0 and 1.

---

## 3. Model Architecture

### YOLO11n (Nano)
| Property        | Value             |
|-----------------|-------------------|
| Architecture    | YOLO11 Nano       |
| Parameters      | ~2.59 Million     |
| GFLOPs          | 6.4               |
| Input Size      | 416 × 416         |
| Pretrained On   | COCO (80 classes) |
| Fine-tuned On   | BDD100K 2k subset |
| Output Classes  | 6                 |

### Why YOLO11n?
- Lightest YOLO11 variant — suitable for CPU-only machines
- Fast inference with acceptable accuracy
- Pretrained weights transfer well to driving datasets

---

## 4. Training Configuration

| Parameter   | Value              | Reason                              |
|-------------|--------------------|------------------------------------|
| `epochs`    | 3                  | Quick run; increase for better mAP |
| `imgsz`     | 416                | Lower RAM than 640                 |
| `batch`     | 4                  | Prevents OOM on CPU                |
| `workers`   | 0                  | Avoids Windows Jupyter multiprocessing bug |
| `cache`     | False              | Doesn't load all images into RAM   |
| `device`    | cpu                | Local machine (no GPU)             |
| `optimizer` | Auto (MuSGD)       | Auto-selected by YOLO              |
| `lr0`       | 0.001              | Auto-tuned                         |

**Weights saved to:**
```
Object\checkpoints\yolo_2k_experiment\weights\best.pt
Object\checkpoints\yolo_2k_experiment\weights\last.pt
```

---

## 5. Code Execution Order

> ⚠️ **Open `final_detection.ipynb` in Jupyter Notebook and run cells in the following order:**

---

### ▶ Step 1 — Cell 1: Verify Dataset
**What it does:** Checks that the `2k` folder is populated correctly.  
**Expected output:**
```
✅ train   images: 1400   labels: 1400
✅ val     images:  400   labels:  400
✅ test    images:  200   labels:  200
data_2k.yaml exists: ✅
```
> Run this **first** — confirms everything is in place before training begins.

---

### ▶ Step 2 — Cell 2: Write data_2k.yaml
**What it does:** Creates/refreshes the YAML config file that tells YOLO where the dataset lives.  
**Expected output:**
```
✅  data_2k.yaml written to: ...Object\data_2k.yaml
```
> Must run **before training** — YOLO will fail without this file.

---

### ▶ Step 3 — Cell 3: Dataset Class
**What it does:** Defines `YOLODataset` class and loads a sample image to verify the pipeline works.  
**Expected output:**
```
YOLODataset loaded: 1400 train images
Sample image shape : (416, 416, 3)   boxes: N
```
> Verifies images are readable and labels are parseable.

---

### ▶ Step 4 — Cell 4: Train YOLO11n *(Most Important)*
**What it does:** Fine-tunes YOLO11n on the 2k subset for 3 epochs.  
**Expected output:**
```
🚀  Starting training on 2k subset ...
Epoch 1/3 ...
Epoch 2/3 ...
Epoch 3/3 ...
✅  Training done in ~XX min.
```
> ⏱ Takes ~20–40 minutes on CPU.  
> ⚠️ Do **not** skip Cells 1 & 2 before this.

---

### ▶ Step 5 — Cell 5: Validate & Print Metrics
**What it does:** Loads `best.pt` and evaluates on the validation set.  
**Expected output:**
```
── Evaluation Metrics ──────────────────
  Precision    : 0.XXXX
  Recall       : 0.XXXX
  mAP@0.5      : 0.XXXX
  mAP@0.5:0.95 : 0.XXXX
── Speed ───────────────────────────────
  Inference / image : XX.XX ms
  Approx FPS        : XX.X
```
> Run **after Cell 4** completes.

---

### ▶ Step 6 — Cell 6: Plot Training Curves
**What it does:** Reads `results.csv` and plots box loss, class loss, DFL loss, mAP, precision, recall.  
**Output:** Saves `training_results.png` to the project folder.  
> Run **after Cell 4** completes.

---

### ▶ Step 7 — Cell 7: Vehicle Tracking on Video
**What it does:** Runs YOLO11n + ByteTrack on `6.mp4`. Counts vehicles crossing two virtual lines.  
**Output:**
- Live display window (press **Q** or **Esc** to stop)
- Saved annotated video: `tracking_output.mp4`
- Printed counts: Vehicles Up / Down

> Can run **independently** at any time — uses pretrained `yolo11n.pt` if `best.pt` is not found yet.

---

### 🗺️ Full Execution Flow

```
Cell 1          Cell 2          Cell 3          Cell 4
Verify    →   Write YAML  →   Dataset    →   TRAIN
Dataset         Config         Class         YOLO11n
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                                 Cell 5     Cell 6     Cell 7
                                Validate    Plot      Tracking
                                Metrics    Graphs    on Video
```

---

## 6. Tracking & Counting Logic

The tracking cell uses **ByteTrack** (via `bytetrack.yaml`) for multi-object tracking.

### Counting Lines
Two horizontal virtual lines drawn across the frame:
| Line       | Y Position     | Color |
|------------|----------------|-------|
| Red Line   | 35% of height  | Red   |
| Blue Line  | 50% of height  | Blue  |

### Direction Logic
| Direction | Condition                              |
|-----------|----------------------------------------|
| **Down**  | Vehicle crosses Red → then Blue line   |
| **Up**    | Vehicle crosses Blue → then Red line   |

### Speed Estimation
```
speed = sqrt((cx_current - cx_prev)² + (cy_current - cy_prev)²)
```
Units: pixels per frame (relative motion indicator)

---

## 7. File Structure

```
Object\
├── final_detection.ipynb      ← Main notebook (run this)
├── data_2k.yaml               ← Dataset config for 2k subset
├── data.yaml                  ← Original 10k dataset config
├── tracker.py                 ← Tracker utility
├── 6.mp4                      ← Input video for tracking
├── tracking_output.mp4        ← Output video (after Cell 7)
├── training_results.png       ← Training plots (after Cell 6)
├── Analysis_Report.md         ← This report
├── 10k\                       ← Original full dataset
│   ├── images\ (train/val/test)
│   └── labels\ (train/val/test)
├── 2k\                        ← Reduced dataset used for training
│   ├── images\ (train/val/test)
│   └── labels\ (train/val/test)
└── checkpoints\
    └── yolo_2k_experiment\
        ├── weights\
        │   ├── best.pt        ← Best model weights
        │   └── last.pt        ← Last epoch weights
        └── results.csv        ← Training metrics per epoch
```

---

## 8. Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: not enough memory` | Batch size / image size too large | Already fixed: `batch=4`, `imgsz=416` |
| `FileNotFoundError: data_2k.yaml` | Cell 2 not run yet | Run **Cell 2** first |
| `FileNotFoundError: best.pt` | Training not done | Run **Cell 4** first |
| Jupyter hangs on training | `workers > 0` on Windows | Already fixed: `workers=0` |
| `Cannot open video: 6.mp4` | Wrong working directory | Path is hardcoded — file must be in `Object\` |

---

*Report generated for Minor Project 2 — YOLO11 Object Detection & Tracking Pipeline*
