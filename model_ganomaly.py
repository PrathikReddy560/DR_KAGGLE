import torch
import torch.nn as nn


def conv_block(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_c, out_c, final=False):
    layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=final)]
    if not final:
        layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
    else:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """128x128x3 -> latent_dim vector (as a 1x1 feature map).
    Reused twice inside Generator: once as Ge (image -> z),
    once as Ee (reconstruction -> z_hat)."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, 64, norm=False),   # 128 -> 64
            conv_block(64, 128),              # 64  -> 32
            conv_block(128, 256),             # 32  -> 16
            conv_block(256, 512),             # 16  -> 8
            nn.Conv2d(512, latent_dim, 8, 1, 0),  # 8 -> 1
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """latent_dim vector -> 128x128x3 reconstruction."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 8, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            deconv_block(512, 256),   # 8  -> 16
            deconv_block(256, 128),   # 16 -> 32
            deconv_block(128, 64),    # 32 -> 64
            deconv_block(64, 3, final=True),  # 64 -> 128
        )

    def forward(self, z):
        return self.net(z)


class Generator(nn.Module):
    """GANomaly generator = Encoder1 -> Decoder -> Encoder2 (E-D-E).
    Returns the reconstruction AND both latent codes, because the gap
    between z and z_hat (not the pixel reconstruction error) is what
    becomes the anomaly score at inference time."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.encoder1 = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.encoder2 = Encoder(latent_dim)

    def forward(self, x):
        z = self.encoder1(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        return x_hat, z, z_hat


class Discriminator(nn.Module):
    """WGAN critic. Deliberately has NO BatchNorm (BatchNorm couples
    samples within a batch, which breaks the per-sample gradient used
    in the WGAN-GP penalty) and NO sigmoid (a Wasserstein critic
    outputs an unbounded real-valued score, not a probability)."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 4, 2, 1),
                nn.InstanceNorm2d(o, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # 128 -> 64
            block(64, 128),   # 64 -> 32
            block(128, 256),  # 32 -> 16
            block(256, 512),  # 16 -> 8
        )
        self.critic = nn.Conv2d(512, 1, 8, 1, 0)  # 8 -> 1 scalar score

    def forward(self, x):
        feat = self.features(x)
        score = self.critic(feat).view(-1)
        return score, feat  # features returned too — needed for feature-matching adv loss


if __name__ == "__main__":
    # quick shape smoke test
    x = torch.randn(4, 3, 128, 128)
    G = Generator(latent_dim=100)
    D = Discriminator()
    x_hat, z, z_hat = G(x)
    score, feat = D(x)
    print("x_hat:", x_hat.shape, "z:", z.shape, "z_hat:", z_hat.shape)
    print("score:", score.shape, "feat:", feat.shape)
    assert x_hat.shape == x.shape
    assert z.shape == z_hat.shape
    print("OK")
