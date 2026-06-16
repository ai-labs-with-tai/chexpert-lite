import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def get_resnet50_model(num_classes=1):
    """
    Returns a ResNet50 model with the specified number of output classes.
    """
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_densenet121_model(num_classes=5, use_dropout=False):
    """
    Returns a DenseNet121 model with the specified number of output classes.
    For multi-label classification (5 diseases).
    Args:
        num_classes: number of output classes
        use_dropout: if True, add Dropout layer before final Linear (for exp1 model)
    """
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    
    if use_dropout:
        # Architecture with Dropout (like exp1 model)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        # Simple Linear layer
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
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
            
    # Handle the case where the saved model has a different number of classes
    # For ResNet (fc layer)
    fc_weight_key = 'fc.weight'
    if fc_weight_key in new_state_dict and hasattr(model, 'fc'):
        out_features = new_state_dict[fc_weight_key].shape[0]
        if model.fc.out_features != out_features:
            print(f"Warning: Model FC output dimension mismatch. Expected {model.fc.out_features}, but weights have {out_features}. Adjusting model.")
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, out_features)
    
    # For DenseNet (classifier layer)
    classifier_weight_key = 'classifier.weight'
    if classifier_weight_key in new_state_dict and hasattr(model, 'classifier'):
        out_features = new_state_dict[classifier_weight_key].shape[0]
        if model.classifier.out_features != out_features:
            print(f"Warning: Model classifier output dimension mismatch. Expected {model.classifier.out_features}, but weights have {out_features}. Adjusting model.")
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, out_features)
            
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def discover_diseases(base_dir):
    """
    Scans the DAM directory to find available diseases.
    Assumes naming convention: {Disease}_latest.pth
    """
    dam_dir = os.path.join(base_dir, "CheXpert_DAM", "Resnet50", "DAM")
    ce_dir = os.path.join(base_dir, "CheXpert_DAM", "Resnet50", "CE")
    
    diseases = set()
    if os.path.exists(dam_dir):
        for filename in os.listdir(dam_dir):
            if filename.endswith("_latest.pth"):
                disease = filename.replace("_latest.pth", "")
                diseases.add(disease)
                
    # Fallback to check CE dir if DAM is missing some
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

def predict_multilabel(model, image_tensor):
    """
    Runs inference for multi-label classification and returns probabilities for all classes.
    Returns a dictionary with disease names and their probabilities.
    """
    disease_names = [
        'Cardiomegaly',
        'Edema', 
        'Consolidation',
        'Atelectasis',
        'Pleural Effusion'
    ]
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply sigmoid for multi-label classification
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    # Return as dictionary
    return {disease_names[i]: float(probs[i]) for i in range(len(disease_names))}


class GradCAM:
    """
    Grad-CAM implementation for visualizing what the model is looking at
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate CAM for a given input
        Args:
            input_tensor: preprocessed image tensor (1, 3, H, W)
            target_class: class index to generate CAM for (None for highest score)
        Returns:
            cam: numpy array of the CAM, normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        if output.dim() == 2 and output.shape[1] == 1:
            # Binary classification (single output)
            target = output[0, 0]
        else:
            # Multi-class
            target = output[0, target_class]
        
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)
        
        # Weight the channels by gradient
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Create CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

def get_gradcam_layer(model):
    """
    Get the appropriate layer for Grad-CAM based on model architecture
    """
    if isinstance(model, models.ResNet):
        # For ResNet, use the last conv layer
        return model.layer4[-1].conv3 if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1].conv2
    elif isinstance(model, models.DenseNet):
        # For DenseNet, use features.denseblock4
        return model.features.denseblock4
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

def apply_gradcam_overlay(original_image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Apply Grad-CAM heatmap overlay on original image
    Args:
        original_image: PIL Image or numpy array (H, W, 3)
        cam: numpy array of CAM (h, w)
        alpha: transparency of overlay (0-1)
        colormap: OpenCV colormap
    Returns:
        PIL Image with overlay
    """
    # Convert PIL to numpy if needed
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Ensure RGB
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
    
    # Resize CAM to match image size
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert CAM to heatmap
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlayed)

def generate_gradcam_visualization(model, image, image_tensor, class_idx=None):
    """
    Generate Grad-CAM visualization for a model prediction
    Args:
        model: trained model
        image: original PIL Image
        image_tensor: preprocessed tensor for model
        class_idx: class index (for multi-class), None for binary
    Returns:
        tuple: (cam_overlay, cam_heatmap, raw_cam)
    """
    try:
        # Get appropriate layer
        target_layer = get_gradcam_layer(model)
        
        # Create Grad-CAM object
        gradcam = GradCAM(model, target_layer)
        
        # Generate CAM
        cam = gradcam.generate_cam(image_tensor, target_class=class_idx)
        
        # Create overlay
        overlay = apply_gradcam_overlay(image, cam, alpha=0.4)
        
        # Create pure heatmap
        h, w = np.array(image).shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(heatmap)
        
        return overlay, heatmap_pil, cam
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None, None, None
