"""EfficientNet-B0 Stage-2 models: MC Dropout baseline and primary VBLL head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def _efficientnet_backbone(use_pretrained):
    """Create the required ImageNet-pretrained backbone without silent fallback."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
    try:
        backbone = efficientnet_b0(weights=weights)
    except Exception as error:
        if use_pretrained:
            raise RuntimeError(
                "ImageNet EfficientNet-B0 weights are required by the Stage-2 methodology. "
                "Enable Kaggle Internet once or attach a cached torchvision weight file."
            ) from error
        raise
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()
    return backbone, in_features


class _EfficientNetFeatures(nn.Module):
    """Shared EfficientNet-B0 feature extractor exposing its final conv block."""

    def __init__(self, use_pretrained):
        super().__init__()
        self.backbone, self.feature_dim = _efficientnet_backbone(use_pretrained)

    def extract_features(self, images):
        features = self.backbone.features(images)
        features = self.backbone.avgpool(features)
        return torch.flatten(features, 1)

    @property
    def gradcam_layer(self):
        return self.backbone.features[-1]


class EfficientNetMCDropout(_EfficientNetFeatures):
    """Method A: Dense + Softmax + Dropout, with dropout retained at inference."""

    def __init__(self, num_classes, dropout, use_pretrained=True):
        super().__init__(use_pretrained)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, images):
        return self.classifier(self.dropout(self.extract_features(images)))

    def deterministic_logits(self, images):
        return self.classifier(self.extract_features(images))

    def head_parameters(self):
        return self.classifier.parameters()


class VariationalBayesianLinear(nn.Module):
    """Mean-field Gaussian Bayesian last layer trained by the ELBO objective."""

    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))
        self.prior_std = float(prior_std)
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))

    @staticmethod
    def _std(rho):
        return F.softplus(rho) + 1e-6

    def posterior_mean(self, features):
        return F.linear(features, self.weight_mu, self.bias_mu)

    def sample_logits(self, features, samples):
        """Sample the last-layer posterior while reusing one backbone feature pass."""
        weight = self.weight_mu.unsqueeze(0) + self._std(self.weight_rho).unsqueeze(0) * torch.randn(
            samples, *self.weight_mu.shape, device=features.device, dtype=features.dtype
        )
        bias = self.bias_mu.unsqueeze(0) + self._std(self.bias_rho).unsqueeze(0) * torch.randn(
            samples, *self.bias_mu.shape, device=features.device, dtype=features.dtype
        )
        return torch.einsum("bi,soi->sbo", features, weight) + bias[:, None, :]

    def kl_divergence(self):
        """Analytic KL[q(w)||N(0, prior_std^2)] for the ELBO regulariser."""
        def normal_kl(mu, rho):
            std = self._std(rho)
            prior_var = self.prior_std ** 2
            return 0.5 * torch.sum((std.square() + mu.square()) / prior_var - 1.0 + 2.0 * (
                math.log(self.prior_std) - torch.log(std)
            ))

        return normal_kl(self.weight_mu, self.weight_rho) + normal_kl(self.bias_mu, self.bias_rho)


class EfficientNetVBLL(_EfficientNetFeatures):
    """Method B: EfficientNet-B0 feature extractor plus a Bayesian last layer."""

    def __init__(self, num_classes, prior_std, use_pretrained=True):
        super().__init__(use_pretrained)
        self.vbll = VariationalBayesianLinear(self.feature_dim, num_classes, prior_std)

    def forward(self, images):
        return self.vbll.posterior_mean(self.extract_features(images))

    def deterministic_logits(self, images):
        return self.forward(images)

    def head_parameters(self):
        return self.vbll.parameters()


def build_stage2_model(method, num_classes, dropout, prior_std, use_pretrained=True):
    if method == "mc_dropout":
        return EfficientNetMCDropout(num_classes, dropout, use_pretrained)
    if method == "vbll":
        return EfficientNetVBLL(num_classes, prior_std, use_pretrained)
    raise ValueError(f"Unknown Stage-2 method: {method}")


def configure_finetuning(model, phase, last_layers=20):
    """Implement head-only -> last-20 -> full-backbone progressive unfreezing."""
    for parameter in model.backbone.features.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head_parameters():
        parameter.requires_grad_(True)
    if phase == "head":
        return
    if phase == "last_20":
        leaves = [
            module for module in model.backbone.features.modules()
            if any(True for _ in module.parameters(recurse=False))
        ]
        for module in leaves[-last_layers:]:
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)
        return
    if phase == "full":
        for parameter in model.backbone.features.parameters():
            parameter.requires_grad_(True)
        return
    raise ValueError(f"Unknown fine-tuning phase: {phase}")


def freeze_nontrainable_batchnorm(model):
    """Keep frozen BatchNorm statistics fixed during progressive fine-tuning."""
    for module in model.backbone.features.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
                module.eval()


def enable_mc_dropout(model):
    """Activate dropout at test time without putting BatchNorm layers in train mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def predictive_distribution(model, images, method, samples):
    """Return predictive mean and per-class posterior/MC variance."""
    if method == "mc_dropout":
        enable_mc_dropout(model)
        probability_samples = torch.stack([torch.softmax(model(images), dim=1) for _ in range(samples)])
    elif method == "vbll":
        model.eval()
        features = model.extract_features(images)  # one backbone pass as specified
        logits = model.vbll.sample_logits(features, samples)
        probability_samples = torch.softmax(logits, dim=-1)
    else:
        raise ValueError(f"Unknown Stage-2 method: {method}")
    return probability_samples.mean(dim=0), probability_samples.var(dim=0, unbiased=False)
