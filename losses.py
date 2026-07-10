import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def contextual_loss(x, x_hat):
    """1 - SSIM as the reconstruction loss. Penalizes structural
    mismatch (edges, texture) rather than raw pixel intensity — more
    appropriate for retinal images than L1/L2, since two fundus images
    can differ in brightness but be structurally identical."""
    return 1 - ssim(x, x_hat, data_range=2.0, size_average=True)  # images are in [-1, 1] -> range 2.0


def encoder_loss(z, z_hat):
    """L2 distance between latent codes before/after reconstruction.
    THIS is the anomaly score at inference — not the pixel reconstruction
    error. A healthy retina's latent code survives the decode/re-encode
    round trip; an abnormal one (never seen in training) doesn't."""
    return F.mse_loss(z, z_hat)


def adversarial_loss(feat_real, feat_fake):
    """Feature matching: push reconstructions to have similar
    discriminator features to real images, instead of using the
    (unstable, easy-to-collapse) raw discriminator score directly."""
    return F.mse_loss(feat_fake, feat_real)


def gradient_penalty(discriminator, real, fake, device):
    """WGAN-GP term: penalizes the critic's gradient norm at random
    interpolations between real & fake for deviating from 1, which is
    how WGAN theory enforces the required 1-Lipschitz constraint."""
    eps = torch.rand(real.size(0), 1, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score, _ = discriminator(interp)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


if __name__ == "__main__":
    from model_ganomaly import Generator, Discriminator

    x = torch.randn(4, 3, 128, 128)
    G, D = Generator(), Discriminator()
    x_hat, z, z_hat = G(x)
    _, feat_real = D(x)
    _, feat_fake = D(x_hat)

    print("contextual:", contextual_loss(x, x_hat).item())
    print("encoder:", encoder_loss(z, z_hat).item())
    print("adversarial:", adversarial_loss(feat_real, feat_fake).item())
    print("grad penalty:", gradient_penalty(D, x, x_hat.detach(), "cpu").item())
    print("OK")
