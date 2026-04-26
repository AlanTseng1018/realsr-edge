# EdgeSR-Lab

Edge-friendly super-resolution research lab.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
```

## Data

`data/DIV2K/` is a junction to the local DIV2K dataset (HR only):
- `DIV2K_train_HR/` — 800 training images
- `DIV2K_valid_HR/` — 100 validation images

## Environment

- Python 3.11
- PyTorch (CUDA 12.4 build)
- GPU: tested on RTX 3060 6GB
