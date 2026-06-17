import torch

from src.disease_info import MULTILABEL_DISEASES


def predict_single_disease(model, image_tensor):
    """Return disease probability from a binary or 2-class model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        if outputs.shape[1] == 1:
            probability = torch.sigmoid(outputs).item()
        else:
            probabilities = torch.softmax(outputs, dim=1)
            probability = probabilities[0][1].item()

    return probability


def predict_all_diseases(model, image_tensor):
    """Return probabilities for all DenseNet121 multi-label diseases."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    return {
        MULTILABEL_DISEASES[i]: float(probabilities[i])
        for i in range(len(MULTILABEL_DISEASES))
    }


def interpret_probability(probability, threshold):
    """Convert a probability into a simple Vietnamese risk label."""
    if probability >= threshold:
        return "Khả năng cao", "🔴", "#ff4444"
    if probability >= threshold * 0.7:
        return "Cần theo dõi", "🟡", "#ffaa00"
    return "Khả năng thấp", "🟢", "#00cc66"


def compare_ce_and_dam(prob_ce, prob_dam):
    """Compare two model probabilities for the same disease."""
    return prob_dam - prob_ce


# Backward-compatible names for old tests or notebooks.
predict = predict_single_disease
predict_multilabel = predict_all_diseases
