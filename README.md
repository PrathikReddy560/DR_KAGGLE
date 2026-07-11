# Two-stage diabetic-retinopathy pipeline

Stage 2 follows the approved methodology: ImageNet-pretrained EfficientNet-B0, five-grade DR classification, LAB CLAHE (`clipLimit=2.0`, `8x8` tiles), 70/15/15 stratified real-image splits, inverse-frequency weighted cross-entropy, and two uncertainty methods.

- **Method A:** Dense + Dropout head, with 25 MC Dropout passes at inference.
- **Method B (primary):** Variational Bayesian Last Layer (VBLL), ELBO training, and 15 posterior weight samples after a single backbone pass.

Both methods train progressively: classifier head, final 20 EfficientNet parameterised layers, then the complete backbone. Checkpoints use validation macro-F1, AMP, gradient clipping, weight decay, label smoothing, balanced sampling, cosine scheduling in the first two phases, and `ReduceLROnPlateau` in full fine-tuning.

## Kaggle cells

Attach APTOS 2019 and enable Internet once so torchvision can fetch/cache ImageNet EfficientNet-B0 weights. Pretrained weights are required by the Stage-2 protocol; the code intentionally does not silently fall back to random initialisation.

```python
%cd /kaggle/working/DR_KAGGLE
!pip install -q -r requirements.txt
%env APTOS_DIR=/kaggle/input/aptos2019-blindness-detection
!python train_ganomaly.py
!python train_effnet.py --method all
!python evaluate_effnet.py --method both --split test
```

The report uses APTOS as primary data plus IDRiD and Messidor-2. Attach their prepared manifests when available; each CSV must contain `path,diagnosis` and optional `id_code`, with diagnosis already mapped to `0..4`. Relative paths are resolved from the CSV location.

```python
%env STAGE2_EXTRA_MANIFESTS=/kaggle/input/idrid-manifest/idrid.csv;/kaggle/input/messidor2-manifest/messidor2.csv
```

If cGAN minority-class images from the report are available, attach a manifest with the same columns. They are added only to training and can never contaminate validation or test splits.

```python
%env STAGE2_SYNTHETIC_MANIFEST=/kaggle/input/dr-cgan/synthetic_manifest.csv
```

## Report-aligned inference

The primary VBLL pipeline applies the locked Stage-1 threshold (`GANomaly >= 0.1982`) and Stage-2 confidence threshold (`>= 0.70`). Only predictions passing both gates are accepted and receive a Grad-CAM overlay from EfficientNet-B0's final convolutional block. Stage-1-normal images return `No DR Detected`; low-confidence Stage-2 images return `Refer to Ophthalmologist`.

```python
!python infer.py --method vbll --images "/kaggle/input/my-images/*.png"
```

Outputs in `/kaggle/working` include separate best checkpoints, persisted splits, per-method CSV predictions, accuracy/precision/recall/F1/AUC/ECE/coverage/rejection reports, accepted-prediction confusion matrices, and accepted-only Grad-CAM overlays.
