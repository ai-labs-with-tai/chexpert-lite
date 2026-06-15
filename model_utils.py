import os
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

def get_resnet50_model(num_classes=1):
    """
    Returns a ResNet50 model with the specified number of output classes.
    """
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_weights(model, weights_path):
    """
    Loads weights into the model, handling potential 'module.' prefixes
    if the model was trained with DataParallel.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle case where model was saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
            
    # Also handle the case where the saved model has a different number of classes in the FC layer.
    # We will detect the size of the weight tensor for the fc layer.
    fc_weight_key = 'fc.weight'
    if fc_weight_key in new_state_dict:
        out_features = new_state_dict[fc_weight_key].shape[0]
        if model.fc.out_features != out_features:
            print(f"Warning: Model FC output dimension mismatch. Expected {model.fc.out_features}, but weights have {out_features}. Adjusting model.")
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, out_features)
            
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def discover_diseases(base_dir):
    """
    Scans the DAM directory to find available diseases.
    Assumes naming convention: {Disease}_latest.pth
    """
    dam_dir = os.path.join(base_dir, "CheXpert_DAM", "DAM")
    ce_dir = os.path.join(base_dir, "CheXpert_DAM", "CE")
    
    diseases = set()
    if os.path.exists(dam_dir):
        for filename in os.listdir(dam_dir):
            if filename.endswith("_latest.pth"):
                disease = filename.replace("_latest.pth", "")
                diseases.add(disease)
                
    # Fallback to check CE dir if DAM is missing some (unlikely based on user prompt, but safe)
    if os.path.exists(ce_dir):
        for filename in os.listdir(ce_dir):
            if filename.startswith("CE_") and filename.endswith(".pth"):
                disease = filename.replace("CE_", "").replace(".pth", "")
                diseases.add(disease)
                
    return sorted(list(diseases))

def preprocess_image(image):
    """
    Preprocesses the image for ResNet50 (standard ImageNet normalization).
    Expects a PIL Image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    return image_tensor

def predict(model, image_tensor):
    """
    Runs inference and returns the probability (assuming binary classification or AUC loss).
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        # If output is [1, 1], apply sigmoid
        if outputs.shape[1] == 1:
            prob = torch.sigmoid(outputs).item()
        else:
            # If output is [1, 2] (CrossEntropy), apply softmax and take class 1
            probs = torch.softmax(outputs, dim=1)
            prob = probs[0][1].item() # Assuming class 1 is positive
    return prob
