from torchvision import transforms


def preprocess_image(image):
    """
    Prepare a chest X-ray image before prediction.

    The model expects a 224x224 RGB tensor normalized with ImageNet statistics,
    because ResNet50 and DenseNet121 use the same input format as ImageNet models.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transform(image).unsqueeze(0)
