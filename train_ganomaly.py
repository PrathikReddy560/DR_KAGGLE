import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from config import ON_KAGGLE, APTOS_DIR, ODIR_DIR, GANomalyConfig as C, SEED
from dataset_ganomaly import HealthyRetinaDataset
from model_ganomaly import Generator, Discriminator
from losses import contextual_loss, encoder_loss, adversarial_loss, gradient_penalty

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Kaggle: {ON_KAGGLE} | Device: {device}")

# ---------------------------------------------------------------------------
# Data — healthy-only pool, split into train/val (both still 100% healthy;
# val is used purely for checkpointing, not for any anomaly/AUC evaluation —
# that needs a separate labeled test set with both healthy AND DR images).
# ---------------------------------------------------------------------------
full_dataset = HealthyRetinaDataset(APTOS_DIR, ODIR_DIR, img_size=C.IMG_SIZE)
n_val = max(1, int(len(full_dataset) * C.VAL_SPLIT))
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)
print(f"Train: {n_train} | Val: {n_val}")

train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                           num_workers=2, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
G = Generator(C.LATENT_DIM).to(device)
D = Discriminator().to(device)
opt_G = Adam(G.parameters(), lr=C.LR, betas=(C.BETA1, C.BETA2))
opt_D = Adam(D.parameters(), lr=C.LR, betas=(C.BETA1, C.BETA2))

best_val_loss = float("inf")

for epoch in range(C.EPOCHS):
    G.train(); D.train()
    running_g, running_d = 0.0, 0.0

    for real in train_loader:
        real = real.to(device)

        # ---- Critic steps (WGAN-GP: train D more than G) ----
        for _ in range(C.N_CRITIC):
            with torch.no_grad():
                fake, _, _ = G(real)
            score_real, _ = D(real)
            score_fake, _ = D(fake)
            gp = gradient_penalty(D, real, fake, device)
            d_loss = score_fake.mean() - score_real.mean() + C.GP_WEIGHT * gp

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

        # ---- Generator step ----
        fake, z, z_hat = G(real)
        _, feat_real = D(real)
        _, feat_fake = D(fake)

        l_adv = adversarial_loss(feat_real, feat_fake)
        l_con = contextual_loss(real, fake)
        l_enc = encoder_loss(z, z_hat)
        g_loss = C.W_ADV * l_adv + C.W_CON * l_con + C.W_ENC * l_enc

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        running_g += g_loss.item()
        running_d += d_loss.item()

    # ---- Validation: encoder loss on held-out healthy images ----
    # (this — not a reconstruction metric — is what the report calls
    # "validation loss", since it's the same quantity used as the
    # anomaly score at inference time)
    G.eval()
    val_enc_loss = 0.0
    with torch.no_grad():
        for real in val_loader:
            real = real.to(device)
            _, z, z_hat = G(real)
            val_enc_loss += encoder_loss(z, z_hat).item()
    val_enc_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{C.EPOCHS} | "
          f"G: {running_g/len(train_loader):.4f} | "
          f"D: {running_d/len(train_loader):.4f} | "
          f"Val Enc Loss: {val_enc_loss:.4f}")

    if val_enc_loss < best_val_loss:
        best_val_loss = val_enc_loss
        torch.save(G.state_dict(), C.CKPT_PATH)
        print(f"  -> New best ({best_val_loss:.4f}), checkpoint saved.")

print(f"Training complete. Best val encoder loss: {best_val_loss:.4f}")
print(f"Best checkpoint: {C.CKPT_PATH}")
