import os
from pathlib import Path


# Use relative path by default so the project can run on another machine.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("CHEXPERT_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()

CHEXPERT_DAM_DIR = PROJECT_ROOT / "CheXpert_DAM"
RESNET50_DAM_DIR = CHEXPERT_DAM_DIR / "Resnet50" / "DAM"
RESNET50_CE_DIR = CHEXPERT_DAM_DIR / "Resnet50" / "CE"
DENSENET121_MODEL_PATH = CHEXPERT_DAM_DIR / "Densenet121" / "best_densenet_exp1.pth"

DEFAULT_THRESHOLD = 0.5
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
