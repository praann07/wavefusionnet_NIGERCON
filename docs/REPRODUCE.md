# Reproducing the experiments

Every experiment is one Kaggle notebook. There is nothing to install: Kaggle's default image already has PyTorch, torchvision, NumPy, Pillow, Matplotlib and tqdm.

## Steps for any notebook

1. On Kaggle choose **Create, New Notebook, File, Import Notebook** and select the `.ipynb` file.
2. **Settings, Accelerator:** GPU T4 x2. **Settings, Internet:** On. Internet is needed once, for `pip install torchinfo` and to download the VGG19 weights used by the perceptual loss. The L1-only notebooks need neither VGG nor internet for the loss.
3. **Add Input** and attach the dataset listed in the table below. The notebook reads it from `/kaggle/input/...`.
4. Run all cells without editing anything. Use **Save Version, Save & Run All** so the run continues if you close the browser.
5. When it finishes, check the printed `TEST PSNR / SSIM` against the table, and compare the `split_proof` JSON it writes with `results/split_proofs/` (the train hash must start with the prefix shown in the README). A run whose split hashes differ is not comparable and should be discarded.

A Kaggle session lasts 12 hours and each run below takes about 10, so there is room for at most one restart. The notebooks do not resume from a checkpoint.

## Full models

| Notebook | Kaggle input | Epochs | Expected TEST (PSNR / SSIM) | Time |
|---|---|---:|---|---|
| `train/gopro_full_model.ipynb` | `gopro-data` and `file-paths-go-pro-entire` | 100 | 28.83 dB / 0.8754 | about 10 h |
| `train/realblurj_full_model.ipynb` | `realblurj-image-deblurring-dataset` | 70 | 29.57 dB / 0.8839 | about 9.7 h |
| `train/hide_full_model.ipynb` | HIDE upload, read from `/kaggle/input/datasets` (edit `data_root` in the config cell to match your upload) | 35 | 28.93 dB / 0.8937 | about 10.5 h |

## Ablations

| Notebook | Change from the full model | Dataset | Expected TEST | Where it was run |
|---|---|---|---|---|
| `ablation/gopro_fixed_fusion.ipynb` | Learned gate replaced by fixed concatenation | GoPro | 28.79 / 0.8748 | local RTX 5060 |
| `ablation/gopro_spatial_only.ipynb` | Wavelet branch and fusion removed | GoPro | 28.69 / 0.8730 | local RTX 5060 |
| `ablation/realblurj_fixed_fusion_part1_interrupted.ipynb` then `..._part2_resumed_from_epoch50.ipynb` | Fixed concatenation; interrupted at epoch 50, resumed with a fresh AdamW state | RealBlur-J | 29.55 / 0.8841 | local RTX 5060 |
| `ablation/realblurj_spatial_only.ipynb` | Wavelet branch and fusion removed | RealBlur-J | 29.53 / 0.8819 | Kaggle |
| `ablation/new_runs/gopro_l1only_loss.ipynb` | L1 loss only | GoPro | 28.28 / 0.8652 | Kaggle |
| `ablation/new_runs/gopro_no_stripattention.ipynb` | StripAttention replaced by identity | GoPro | 28.75 / 0.8741 | Kaggle |
| `ablation/new_runs/realblurj_l1only_loss.ipynb` | L1 loss only | RealBlur-J | 29.18 / 0.8775 | Kaggle |
| `ablation/new_runs/realblurj_no_stripattention.ipynb` | StripAttention replaced by identity | RealBlur-J | 29.57 / 0.8836 | Kaggle |
| `ablation/new_runs/realblurj_seed7_full_model.ipynb` | Training seed 7 (split seed stays 42) | RealBlur-J | 29.57 / 0.8841 | Kaggle |

The local runs used the same code as the Kaggle notebooks with paths pointed at a local copy of the data, so their saved outputs show local Windows paths.

Each ablation changes exactly one thing. The diff against the corresponding full-model notebook is 1 line for No StripAttention, 3 small edits for L1-only, and 2 lines for the seed run.

## The seed run

`realblurj_seed7_full_model.ipynb` sets `train_seed = 7` for training but builds the train/validation split from `split_seed = 42`. This is deliberate. Changing the split seed would change which images are in each set and break the identical-split guarantee, so the two seeds are kept separate.

## Checking the splits

```bash
python scripts/verify_splits.py
```

This confirms that all runs on a dataset share identical hashes and that they match the published prefixes. It reads only the JSON files in `results/split_proofs/`, so it needs no dataset and no GPU.

## Note on the checkpoint rule

The best checkpoint is chosen by validation PSNR, evaluated at epoch 1, every 10 epochs, and at the final epoch. The test set is run once with that checkpoint after training ends. The test set never influences model selection.
