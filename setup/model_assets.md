# Model Assets

## Required

1. XPart model bundle
   - Source: `tencent/Hunyuan3D-Part`
   - Download mode: snapshot
  - Destination: `./models/tencent/Hunyuan3D-Part`

2. P3-SAM checkpoint
   - Source: `tencent/Hunyuan3D-Part`
   - File: `p3sam/p3sam.safetensors`
  - Destination: `./models/p3sam/weights/p3sam/p3sam.safetensors`

3. Sonata checkpoints
   - Source: `facebook/sonata`
   - Files:
     - `sonata.pth`
     - `sonata_small.pth`
     - `sonata_linear_prob_head_sc.pth`
  - Destination: `./models/sonata/ckpt/`

## Optional

1. Sonata demo data
   - Source: `pointcept/demo` (dataset repo)
   - Files:
     - `sample1.npz`
     - `sample1_high_res.npz`
     - `sample1_dino.npz`
  - Destination: `./models/sonata/data/`

## Manual Download Script

- Script path: `scripts/download_models.py`
- Dry run:
  - `python scripts/download_models.py --dry-run`
- Full download:
  - `python scripts/download_models.py`
- Include optional Sonata demo data:
  - `python scripts/download_models.py --include-sonata-demo-data`
- With token:
  - `python scripts/download_models.py --hf-token <YOUR_HF_TOKEN>`
