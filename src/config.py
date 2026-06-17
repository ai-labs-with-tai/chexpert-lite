import os


def find_project_root():
    """Find the folder that contains both CheXpert_GUI and CheXpert_DAM."""
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        os.path.abspath(os.path.join(os.getcwd(), "..")),
        os.path.abspath(os.getcwd()),
        r"D:\CheXpert_Lite",
    ]

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "CheXpert_DAM")):
            return candidate

    return candidates[0]


PROJECT_ROOT = find_project_root()

CHEXPERT_DAM_DIR = os.path.join(PROJECT_ROOT, "CheXpert_DAM")
RESNET50_DAM_DIR = os.path.join(CHEXPERT_DAM_DIR, "Resnet50", "DAM")
RESNET50_CE_DIR = os.path.join(CHEXPERT_DAM_DIR, "Resnet50", "CE")
DENSENET121_MODEL_PATH = os.path.join(CHEXPERT_DAM_DIR, "Densenet121", "best_densenet_exp1.pth")

DEFAULT_THRESHOLD = 0.5
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
