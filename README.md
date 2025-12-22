# Qwen-Image-Layered (Implementation Plan)

Local project scaffold to replicate and extend the multilayer diffusion model described in [Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition](https://arxiv.org/html/2512.15603v1). The goal is to build a trainable and evaluable pipeline that learns to decompose raster images into semantically disentangled RGBA layers, leveraging the high-quality multilayer dataset already available on this machine.

## Local dataset hooks
- Rendered RGBA samples: `/home/ubuntu/jjseol/layer_data/inpainting_250k_subset_rendered`
- Layout JSON with component metadata/descriptions: `/home/ubuntu/jjseol/layer_data/inpainting_250k_subset`
- Visualization notebook reference: `/home/ubuntu/jjseol/data/test_lica_api copy.ipynb` (shows how to load backgrounds, components, and masks)

## Directory layout
- `src/` model, data pipelines, training and evaluation scripts  
- `configs/` experiment configs (model hyperparams, data paths, training stages)  
- `notebooks/` exploratory notebooks for data inspection and qualitative evaluation  
- `scripts/` CLI utilities (data prep, evaluation, visualization, training entry)  
- `docs/` notes on methodology, ablations, and metrics  

## How to set up
```bash
cd /home/ubuntu/qwen-image-layered
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # adjust torch build if needed (cuda/rocm/cpu)
```

## Data loader scaffold (default)
- Uses the local layered dataset:
  - rendered RGBA: `/home/ubuntu/jjseol/layer_data/inpainting_250k_subset_rendered`
  - layout JSON: `/home/ubuntu/jjseol/layer_data/inpainting_250k_subset`
- Dataset class: `src/data/multilayer_dataset.py`
- Quick check: `python scripts/dataset_sanity_check.py --max-samples 2 --batch-size 1`

### Bucketed RGBA layers for VAE training
- Script: `src/data_generation/prepare_rgba_buckets.py`
- Generates RGBA components **and** their composites, resized via the bucket strategy (rounded multiples of 32, capped max side/pixels).
- Example (train 1k / val 50):
  ```bash
  conda activate training
  PYTHONPATH=. python src/data_generation/prepare_rgba_buckets.py \
      --output-root data/rgba_layers \
      --train-count 1000 \
      --val-count 50
```
- Output layout: `data/rgba_layers/{split}/{bucket}/sample_compXXX.png` and `.../{bucket}/sample_composite.png`, manifest at `metadata/manifest.json`.

### Dataloader for generated data
- `src/data_generation/rgba_component_dataset.py` provides `RgbaComponentDataset` + `create_component_dataloader`.
  ```python
  from src.data_generation import create_component_dataloader

  train_loader = create_component_dataloader(
      root_dir="data/rgba_layers",
      split="train",
      batch_size=16,
      num_workers=4,
  )
  batch = next(iter(train_loader))
  component = batch["component"]      # (B, 4, H, W)
  composite = batch["composite"]      # (B, 4, H, W)
  ```

## Training flow scaffold (paper-aligned)
- Entry: `python scripts/train.py --config configs/default.yaml`
- Stages (to be implemented):
  - `rgba_vae`: shared latent VAE for RGB/RGBA
  - `decompose`: VLD-MMDiT variable-layer decomposition
  - `refine`: task-specific editing refinement
- Default config now consumes the bucketed dataset (`data/rgba_layers`), so each batch contains both a single component layer and the corresponding composite image.
- After every epoch, the RGBA-VAE stage runs validation on the bucketed `val` split:
  - Reconstructs composites, composites them over white background, reports mean PSNR.
  - Saves a grid to `outputs/val/val_recon_epoch_{epoch}.png` for quick inspection.
- Training now uses `Accelerator` (bf16 mixed precision by default). Launch with:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /home/ubuntu/miniconda3/envs/training/bin/python -m accelerate.commands.launch \
      --num_processes=2 scripts/train.py --config configs/default.yaml
  ```
  (Adjust GPU list/process count for your setup.)

## Immediate next steps
- Fill in RGBA-VAE architecture and reconstruction losses.
- Add VLD-MMDiT decomposition head with order-aware DTW/layer-merging loss.
- Implement evaluation: RGB L1 (alpha-weighted), Alpha soft IoU, PSNR/SSIM/rFID/LPIPS for reconstruction, plus qualitative panels.

## Notes on RGBA-VAE init (per paper)
- Inspired by AlphaVAE and Qwen-Image-Layered §3.1: first encoder conv and last decoder conv are widened to four channels.
- Use the provided conversion script to adapt the official Qwen-Image VAE:
  ```bash
  PYTHONPATH=. python scripts/convert_qwen_vae_to_rgba.py \
      --source Qwen/Qwen-Image-1.0 \
      --subfolder vae \
      --output-dir checkpoints/rgba_vae_init
  ```
  (Any Hugging Face repo or local directory containing the original RGB VAE works.)
- Set `model.rgb_checkpoint` to the converted weights; the alpha-channel path starts from zeros (bias via `alpha_bias_init`).
- RGB inputs automatically receive α=1 during training so both RGB/RGBA samples share the latent space.

## Paper-aligned milestones
- **RGBA-VAE**: shared latent space for RGB/RGBA, evaluate on AIM-500-style reconstruction.
- **Variable Layers Decomposition (VLD-MMDiT)**: support variable-length layer outputs, including order-aware training.
- **Multi-stage training**: curriculum from text-to-image init → decomposition fine-tuning → task-specific refinement.
- **Evaluation**: Crello-style protocol (order-aware DTW, layer merging), plus in-house metrics on the local dataset.

## Usage (initial)
- Create and activate a venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps (to be pinned in `requirements.txt` soon): `pip install torch diffusers transformers accelerate scipy numpy pillow matplotlib pyyaml tqdm`
- Start exploring data: copy the existing visualization patterns into `notebooks/` and point to the dataset paths above.

