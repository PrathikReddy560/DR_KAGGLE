"""Grad-CAM on the final EfficientNet-B0 convolutional block for accepted cases."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class EfficientNetGradCAM:
    """Compute Grad-CAM from gradients of the predicted score at features[-1]."""

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._forward_handle = model.gradcam_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output
        # Registering directly on the tensor avoids full-module backward hooks
        # interacting with torchvision's in-place EfficientNet activations.
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients):
        self.gradients = gradients

    def remove(self):
        self._forward_handle.remove()

    def generate(self, image_tensor, class_index):
        """Return a normalised heatmap for a single preprocessed input image."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model.deterministic_logits(image_tensor)
        logits[:, int(class_index)].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not receive final convolutional activations.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        heatmap -= heatmap.min()
        maximum = heatmap.max()
        return heatmap / maximum if maximum > 0 else np.zeros_like(heatmap)


def save_gradcam_overlay(original_image, heatmap, output_path, alpha=0.42):
    """Overlay a red-hot Grad-CAM map on the unmodified retinal image."""
    if not isinstance(original_image, Image.Image):
        original_image = Image.open(original_image).convert("RGB")
    original = np.asarray(original_image.convert("RGB"))
    resized = cv2.resize(heatmap.astype(np.float32), (original.shape[1], original.shape[0]))
    coloured_bgr = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    coloured_rgb = cv2.cvtColor(coloured_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 1.0 - alpha, coloured_rgb, alpha, 0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    return str(output_path)
