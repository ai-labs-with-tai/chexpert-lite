import os

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import DENSENET121_MODEL_PATH, RESNET50_CE_DIR, RESNET50_DAM_DIR


def get_resnet50_model(num_classes=1):
    """Create a ResNet50 model for one disease prediction."""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def get_densenet121_model(num_classes=5, use_dropout=False):
    """Create a DenseNet121 model for multi-label disease prediction."""
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features

    if use_dropout:
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes),
        )
    else:
        model.classifier = nn.Linear(num_ftrs, num_classes)

    return model


def load_weights(model, weights_path):
    """Load saved PyTorch weights and remove DataParallel prefixes if needed."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value

    fc_weight_key = "fc.weight"
    if fc_weight_key in new_state_dict and hasattr(model, "fc"):
        out_features = new_state_dict[fc_weight_key].shape[0]
        if model.fc.out_features != out_features:
            print(
                f"Warning: Model FC output dimension mismatch. Expected {model.fc.out_features}, "
                f"but weights have {out_features}. Adjusting model."
            )
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, out_features)

    classifier_weight_key = "classifier.weight"
    if classifier_weight_key in new_state_dict and hasattr(model, "classifier"):
        out_features = new_state_dict[classifier_weight_key].shape[0]
        if model.classifier.out_features != out_features:
            print(
                f"Warning: Model classifier output dimension mismatch. Expected {model.classifier.out_features}, "
                f"but weights have {out_features}. Adjusting model."
            )
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, out_features)

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def discover_diseases(base_dir=None):
    """Find disease names from available CE and DAM weight files."""
    dam_dir = RESNET50_DAM_DIR
    ce_dir = RESNET50_CE_DIR

    if base_dir is not None:
        dam_dir = os.path.join(base_dir, "CheXpert_DAM", "Resnet50", "DAM")
        ce_dir = os.path.join(base_dir, "CheXpert_DAM", "Resnet50", "CE")

    diseases = set()
    if os.path.exists(dam_dir):
        for filename in os.listdir(dam_dir):
            if filename.endswith("_latest.pth"):
                diseases.add(filename.replace("_latest.pth", ""))

    if os.path.exists(ce_dir):
        for filename in os.listdir(ce_dir):
            if filename.startswith("CE_") and filename.endswith(".pth"):
                diseases.add(filename.replace("CE_", "").replace(".pth", ""))

    return sorted(diseases)


def get_resnet50_weight_paths(disease, base_dir=None):
    """Return the DAM and CE weight paths for one disease."""
    dam_dir = RESNET50_DAM_DIR
    ce_dir = RESNET50_CE_DIR

    if base_dir is not None:
        dam_dir = os.path.join(base_dir, "CheXpert_DAM", "Resnet50", "DAM")
        ce_dir = os.path.join(base_dir, "CheXpert_DAM", "Resnet50", "CE")

    dam_path = os.path.join(dam_dir, f"{disease}_latest.pth")
    ce_path_1 = os.path.join(ce_dir, f"CE_{disease}.pth")
    ce_path_2 = os.path.join(ce_dir, f"CE_{disease.replace('_', ' ')}.pth")
    ce_path = ce_path_1 if os.path.exists(ce_path_1) else ce_path_2

    return ce_path, dam_path


def load_ce_model(disease, base_dir=None):
    ce_path, _ = get_resnet50_weight_paths(disease, base_dir)
    model = get_resnet50_model()
    return load_weights(model, ce_path)


def load_dam_model(disease, base_dir=None):
    _, dam_path = get_resnet50_weight_paths(disease, base_dir)
    model = get_resnet50_model()
    return load_weights(model, dam_path)


def load_densenet121_model(base_dir=None):
    model_path = DENSENET121_MODEL_PATH
    if base_dir is not None:
        model_path = os.path.join(base_dir, "CheXpert_DAM", "Densenet121", "best_densenet_exp1.pth")

    model = get_densenet121_model(num_classes=5, use_dropout=True)
    return load_weights(model, model_path)
