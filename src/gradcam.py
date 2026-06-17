import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image


class GradCAM:
    """Grad-CAM implementation for visualizing what the model is looking at."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """Generate a normalized class activation map from model gradients."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()

        if output.dim() == 2 and output.shape[1] == 1:
            target = output[0, 0]
        else:
            target = output[0, target_class]

        target.backward()

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_gradcam_layer(model):
    """Choose the last useful convolution layer for ResNet50 or DenseNet121."""
    if isinstance(model, models.ResNet):
        return model.layer4[-1].conv3 if hasattr(model.layer4[-1], "conv3") else model.layer4[-1].conv2
    if isinstance(model, models.DenseNet):
        return model.features.denseblock4
    raise ValueError(f"Unsupported model type: {type(model)}")


def apply_gradcam_overlay(original_image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay a Grad-CAM heatmap on the original image."""
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)

    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)

    height, width = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (width, height))

    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlayed)


def generate_gradcam_visualization(model, image, image_tensor, class_idx=None):
    """Create Grad-CAM overlay, heatmap image, and raw CAM values."""
    try:
        target_layer = get_gradcam_layer(model)
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(image_tensor, target_class=class_idx)

        overlay = apply_gradcam_overlay(image, cam, alpha=0.4)

        height, width = np.array(image).shape[:2]
        cam_resized = cv2.resize(cam, (width, height))
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(heatmap)

        return overlay, heatmap_pil, cam
    except Exception as error:
        print(f"Error generating Grad-CAM: {error}")
        return None, None, None
