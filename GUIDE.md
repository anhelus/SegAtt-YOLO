# SegAtt-YOLO — Training + XAI (Two Steps)

## 1. TRAIN

### Config: `train_config.yaml`
```yaml
models:
  - yolo11          # base name (scale 'n' added automatically)
  - yolo11_gam
  - yolo11_simam
  # ...
datasets:
  - name: tomatoes
    path: data/tomatoes/yolotomato_bboxes.yaml
    project: runs/tomatoes
training:
  epochs: 100
  batch: 16
  imgsz: 640
  device: 0          # GPU index, or 'cpu'
```

### Run
```powershell
# Single model, single dataset, custom epochs
python -m scripts.batch_train \
  --config train_config.yaml \
  --model yolo11 \
  --epochs 10 \
  --data data/tomatoes/yolotomato_bboxes.yaml \
  --verbose

# All models from config
python -m scripts.batch_train --config train_config.yaml --epochs 100 --verbose
```

### Output
```
runs/tomatoes/yolo11/weights/best.pt    # <-- folder name = model from config (no scale suffix)
runs/tomatoes/yolo11_gam/weights/best.pt
...
```

---

## 2. FIX MODEL FOLDER NAMES (required for XAI)

XAI config expects **scale suffix** (`yolo11n`, `yolo11n-gam`), but training uses **base names** (`yolo11`, `yolo11_gam`).

```powershell
# After training, rename each model folder to add 'n' suffix:
Copy-Item -Recurse "runs/tomatoes/yolo11"        "runs/tomatoes/yolo11n"
Copy-Item -Recurse "runs/tomatoes/yolo11_gam"    "runs/tomatoes/yolo11n-gam"
Copy-Item -Recurse "runs/tomatoes/yolo11_simam"  "runs/tomatoes/yolo11n-simam"
# ...repeat for all trained models and datasets
```

---

## 3. XAI (Explainable AI)

### Config: `xai_config.yaml`
```yaml
methods:
  - eigencam
  - activation

common:
  output_base: reports/xai
  conf_threshold: 0.25
  device: null
  target_location: both

datasets:
  tomatoes:
    source: data/tomatoes/val/images
    model_base: runs/tomatoes
    models:
      - yolo11n          # MUST match folder names above
      - yolo11n-gam
      - yolo11n-simam
      # ...
```

### Run
```powershell
# All methods, all models for one dataset
python -m scripts.batch_xai --config xai_config.yaml --dataset tomatoes --verbose

# Single method + model
python -m scripts.batch_xai --config xai_config.yaml --dataset tomatoes --model yolo11n --method eigencam --verbose

# Single image (direct script)
python -m scripts.xai_predict \
  --model runs/tomatoes/yolo11n/weights/best.pt \
  --source data/tomatoes/val/images \
  --method eigencam \
  --output reports/xai/tomatoes/yolo11n/eigencam \
  --verbose
```

### Methods
| Method | Speed | Gradients | Class-specific |
|--------|-------|-----------|----------------|
| `eigencam` | Fast | No | No |
| `gradcam` | Medium | Yes | Yes |
| `gradcam++` | Medium | Yes | Yes |
| `ss-gradcam++` | Slow | Yes | Yes |
| `activation` | Fast | No | No |

### Output
```
reports/xai/tomatoes/yolo11n/
├── eigencam/
│   ├── img1_eigencam.jpg      # heatmap + boxes
│   └── npy/                   # raw arrays (if --save-npy)
├── activation/
└── GAM_0/                     # attention maps from GAM modules
    └── img1/
        └── spatial_attention.jpg
```

---

## Common Issues

| Error | Fix |
|-------|-----|
| `No such file: bus.jpg` | `curl -L https://ultralytics.com/images/bus.jpg -o segatt_yolo/assets/bus.jpg` |
| `Model not found` | Folder name mismatch → run the **Fix step above** |
| `CUDA device=0 invalid` | Install PyTorch CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| `0 epochs trained` | Dataset empty or wrong `device` (see CUDA fix) |

---

## One-Liner (PowerShell)
```powershell
# Train
python -m scripts.batch_train --config train_config.yaml --model yolo11 --epochs 10 --data data/tomatoes/yolotomato_bboxes.yaml --verbose

# Fix folder name
Copy-Item -Recurse "runs/tomatoes/yolo11" "runs/tomatoes/yolo11n"

# XAI
python -m scripts.batch_xai --config xai_config.yaml --dataset tomatoes --verbose
```

Repeat for each model (`yolo11_gam`, `yolo11_simam`, etc.) and dataset.