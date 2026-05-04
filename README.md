# Section 2: Data Analysis and Machine Learning

---

## Part 1: Multi-Task CNN — Object Detection & Depth Estimation

### Dataset: Cityscapes

Left-side 8-bit LDR camera frames used as input. Left/right perspectives are nearly identical so only one side is taken to reduce complexity and avoid overfitting.
16-bit images excluded due to memory cost. Depth ground truth provided as 16-bit disparity maps. Grayscale label masks (`gtFine_labelIds`) used over JSON annotations for simpler file handling.

Cityscapes organises images by city — this is unneeded for object detection task.
1000 images are sampled from all cities regardless of city origin, and flattened into train/val/test folders to prevent the model learning city-specific patterns. 
No official test set is provided becuase CityScape doesn't want users to 'cheat' the results, so images are randomly sampled from train and val sets.

| Class      | Label ID |
|------------|----------|
| Person     | 24       |
| Car        | 26       |
| Truck      | 27       |
| Motorcycle | 32       |
| Bicycle    | 33       |

---

### Pre-processing

```python
IMG_SRC  = 'leftImg8bit_data/leftImg8bit'
GT_SRC   = 'gtFine_trainvaltest_png_instances/gtFine'
DISP_SRC = 'disparity_trainvaltest/disparity'
OUT_DIR  = 'Cityscapes_Lean'
SIZE     = (512, 256)  # 2:1 aspect ratio maintained
LIMITS   = {'train': 1500, 'val': 500, 'test': 500}
```

Images resized from `(1024, 2048)` to `(512, 256)`. `INTER_NEAREST` interpolation is mandatory for masks — other methods average neighbouring pixels and corrupt integer class IDs.

```python
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, SIZE, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(mask_out, f'{base_id}.png'), mask)

# ImageNet normalisation
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

---

### Model: Multi-Head CNN

Shared encoder feeding two decoder heads — object detection and depth estimation. Minimises VRAM and allows shared feature learning between tasks.

| Stage                  | Output Shape      | Features Learned                        |
|------------------------|-------------------|-----------------------------------------|
| Input                  | [B, 3, 256, 512]  | Raw image                               |
| Encoder 1              | [B, 32, 128, 256] | Edges, colour gradients                 |
| Encoder 2              | [B, 64, 64, 128]  | Shape outlines                          |
| Encoder 3              | [B, 128, 32, 64]  | Shape information                       |
| Encoder 4              | [B, 256, 16, 32]  | Semantic object representation          |
| Encoder 5 (Bottleneck) | [B, 512, 8, 16]   | Shared feature space for both heads     |
| Spatial Pool           | [B, 512, 4, 8]    | Spatial awareness for bounding boxes    |

**Detection Head:** `[B, 16384] → [B, 25]` — object presence per class  
**Depth Head:** `[B, 512, 8, 16] → [B, 1, 256, 512]` — per-pixel depth

---

### Training

```python
optimizer    = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
mse          = nn.MSELoss()
epochs       = 20
lamda_box    = 5.0
lambda_depth = 1.0
```

**Issues encountered during training:**

**Gradient accumulation** — PyTorch accumulates gradients by default, causing gradient explosion with batch size 32. Fixed by zeroing each step:
```python
optimizer.zero_grad()
```

**Bounding box loss** — PyTorch MSE averaged over all classes including absent objects. Manual masked MSE implemented:
```python
mask      = output_v[:, :, 0].unsqueeze(-1).expand_as(detection_out_v[:, :, 1:])
bbox_loss = (mask * (detection_out_v[:, :, 1:] - output_v[:, :, 1:]) ** 2).sum() / (mask.sum() + 1e-6)
l_det     = pres_loss + lamda_box * bbox_loss
```

**Depth loss** — skipped when no object detected:
```python
if depth.sum() > 0:
    l_dep = mse(depth_out, depth) * lambda_depth
else:
    l_dep = torch.tensor(0.0)
(l_det + l_dep).backward()
```

**NumPy on GPU** — numpy must run on CPU:
```python
detect_np = detection_out_v.view(-1, num_classes, 5).cpu().numpy()
```

Best validation loss: `0.2882`

<details>
<summary>Epoch log</summary>

```
Epoch 01/20 | Train Det: 0.3749 Dep: 0.1137 | Val Det: 0.3149 Dep: 0.1020
Epoch 02/20 | Train Det: 0.3339 Dep: 0.0955 | Val Det: 0.3089 Dep: 0.0957
Epoch 03/20 | Train Det: 0.3128 Dep: 0.0878 | Val Det: 0.2923 Dep: 0.0892
Epoch 04/20 | Train Det: 0.3005 Dep: 0.0822 | Val Det: 0.2854 Dep: 0.0803
Epoch 05/20 | Train Det: 0.2893 Dep: 0.0770 | Val Det: 0.2789 Dep: 0.0685
Epoch 06/20 | Train Det: 0.2831 Dep: 0.0723 | Val Det: 0.2849 Dep: 0.0712
Epoch 07/20 | Train Det: 0.2782 Dep: 0.0677 | Val Det: 0.2979 Dep: 0.0677
Epoch 08/20 | Train Det: 0.2742 Dep: 0.0641 | Val Det: 0.2686 Dep: 0.0587
Epoch 09/20 | Train Det: 0.2690 Dep: 0.0605 | Val Det: 0.2683 Dep: 0.0564
Epoch 10/20 | Train Det: 0.2665 Dep: 0.0577 | Val Det: 0.2660 Dep: 0.0513
Epoch 11/20 | Train Det: 0.2643 Dep: 0.0552 | Val Det: 0.2656 Dep: 0.0482
Epoch 12/20 | Train Det: 0.2580 Dep: 0.0523 | Val Det: 0.2616 Dep: 0.0519
Epoch 13/20 | Train Det: 0.2570 Dep: 0.0507 | Val Det: 0.2681 Dep: 0.0493
Epoch 14/20 | Train Det: 0.2564 Dep: 0.0489 | Val Det: 0.2615 Dep: 0.0480
Epoch 15/20 | Train Det: 0.2523 Dep: 0.0463 | Val Det: 0.2791 Dep: 0.0458
Epoch 16/20 | Train Det: 0.2496 Dep: 0.0447 | Val Det: 0.2549 Dep: 0.0463
Epoch 17/20 | Train Det: 0.2471 Dep: 0.0435 | Val Det: 0.2541 Dep: 0.0410
Epoch 18/20 | Train Det: 0.2437 Dep: 0.0425 | Val Det: 0.2780 Dep: 0.0407
Epoch 19/20 | Train Det: 0.2424 Dep: 0.0417 | Val Det: 0.2741 Dep: 0.0415
Epoch 20/20 | Train Det: 0.2418 Dep: 0.0399 | Val Det: 0.2524 Dep: 0.0358
```
</details>

---

### Results

**Detection** failed at both IoU thresholds (0.5 and 0.2). Precision, recall and F1 all `0.000` for every class. Root cause identified from visualisation: the model cannot handle multiple objects of the same class — a single bounding box stretches across the full image to enclose all instances simultaneously.

**Depth Estimation (base model, `bbox_loss = 5.0`):**

| Metric    | Value    | Description                          |
|-----------|----------|--------------------------------------|
| RMSE      | 12.617 m | Average depth error in metres        |
| RMSE log  | 40.28%   | Scale-invariant log-space error      |
| Abs Rel   | 35.41%   | Mean absolute relative error         |
| δ < 1.25  | 31.7%    | % pixels within 25% of ground truth |
| δ < 1.25² | 67.3%    | % pixels within 50% of ground truth |
| δ < 1.25³ | 90.3%    | % pixels within 95% of ground truth |

![Bounding Box Visualisation](https://github.com/user-attachments/assets/b859610e-328f-4347-b3e9-80fd5beac424)

---

### Model Variation

Modifications applied:
- Dropout added to encoder blocks
- `bbox_loss` weight increased from `5.0` to `10.0`

Training loss ~10% worse than base model. Depth pixel accuracy improved slightly. Best validation loss: `0.3965`

**Depth Estimation (improved model, `bbox_loss = 10.0`):**

| Metric    | Value    | Description                          |
|-----------|----------|--------------------------------------|
| RMSE      | 13.114 m | Average depth error in metres        |
| RMSE log  | 41.30%   | Scale-invariant log-space error      |
| Abs Rel   | 32.99%   | Mean absolute relative error         |
| δ < 1.25  | 60.2%    | % pixels within 25% of ground truth |
| δ < 1.25² | 80.7%    | % pixels within 50% of ground truth |
| δ < 1.25³ | 90.1%    | % pixels within 95% of ground truth |

---

## Part 2: LSTM — Vehicle Trajectory Prediction

### Dataset: NGSIM US-101

Roadside camera tracking vehicle positions by vehicle ID, filtered to US-101 road only. `Local_X` (lateral lane position) and `Local_Y` (longitudinal motorway position) used for trajectory prediction. All other columns discarded.

**Sampling rate:** 10 Hz  
**Sequence:** 30 previous frames → predict 10 future frames  
**Split by vehicle ID** to prevent data leakage:

```
Vehicles — Train: 400 | Val: 100 | Test: 100
Total valid vehicles: 2847
```

---

### Pre-processing

**Noise filtering** — Savitzky-Golay filter applied due to camera shake noise in NGSIM:
```python
def smooth_track(group):
    group = group.copy()
    if len(group) >= 11:
        group['Local_X'] = savgol_filter(group['Local_X'], window_length=11, polyorder=2)
        group['Local_Y'] = savgol_filter(group['Local_Y'], window_length=11, polyorder=2)
    return group
```

**Short track filtering** — vehicles with fewer than 100 frames removed:
```python
min_frames = 100
valid_ids  = df.groupby('Vehicle_ID').filter(lambda g: len(g) >= min_frames)['Vehicle_ID'].unique()
valid_ids  = np.sort(valid_ids)
```

**Split:**
```python
train_ids = valid_ids[:400]
val_ids   = valid_ids[400:500]
test_ids  = valid_ids[500:600]
```

Sliding window with stride of 5 chosen — stride 10 was too sparse, no stride caused >40 minute training runs. Each vehicle's trajectory is segmented at frame gaps (when leaving camera view) and shifted to its own origin. Standard scaler normalisation applied to handle large numerical discrepancy between `Local_X` and `Local_Y` ranges.

---

### Model: LSTM Encoder-Decoder

```python
model = LSTMTrajectory(input_size=2, hidden_size=128, num_layers=2).to(device)
```

| Layer    | Input         | Output        | Purpose                                      |
|----------|---------------|---------------|----------------------------------------------|
| Encoder  | (b, 30, 2)    | (b, 30, 128)  | Processes 30 observed frames                 |
| Decoder  | (b, 1, 2)     | (b, 1, 128)   | Autoregressively predicts next position      |
| FC Layer | (b, 10, 128)  | (b, 10, 2)    | Maps hidden states to (x, y) coordinates    |

```python
self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
self.decoder = nn.LSTM(2, hidden_size, num_layers, batch_first=True, dropout=0.3)
self.fc = nn.Sequential(
    nn.Linear(hidden_size, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
```

2 hidden layers chosen on the assumption that significant lane changes are infrequent in motorway driving. Layer count increased to 5 for the model variation.

---

### Training & Results

Adam optimiser, `lr=1e-3`, `weight_decay=1e-4`. Learning rate scheduler halves every 5 epochs if loss plateaus. Early stopping after 20 epochs of no improvement. MSE loss function.

```
ADE:  31.956 m  — mean L2 distance over all predicted frames
FDE:  33.334 m  — L2 distance at final predicted frame
RMSE: 34.686 m  — RMS error over all predicted trajectory points
```

Model generalises well to overall traffic flow shape but fails to accurately predict individual vehicle lane positions, consistent with 2 hidden layers being insufficient for learning lateral lane-change behaviour.

**Individual vehicle trajectory:**  
![Individual Vehicle Trajectory](https://github.com/user-attachments/assets/c1a80f9e-d113-4984-a56b-b5acf5f689bb)

**Ground truth simulation:**  
![Ground Truth](https://github.com/user-attachments/assets/5a528551-b715-4aac-a206-a695e5be074d)

**Predicted trajectory:**  
![Predicted Trajectory](https://github.com/user-attachments/assets/85cce307-a66c-42dd-b7ea-eafb46aff04b)
