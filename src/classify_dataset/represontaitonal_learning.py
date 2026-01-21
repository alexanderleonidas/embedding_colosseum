import os
import pickle
from functools import partial
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from dataset_fingerprinter import DatasetFingerprinter
from torch.utils.data import ConcatDataset, DataLoader
from vae import UniversalVAE, vae_loss

from dataset.DataManager import DataManager
from dataset.multimodal_dataset import MultiModalImageDataset


def custom_collate_fn(batch, max_channels: int = 16):
    """
    Custom collate function that pads all tensors to max_channels before batching.

    Args:
        batch: List of samples from dataset
        max_channels: Target number of channels

    Returns:
        Batched tensors with a uniform channel dimension
    """
    # Separate images and labels if present
    if isinstance(batch[0], tuple):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        has_labels = True
    else:
        images = batch
        has_labels = False

    # Pad all images to max_channels
    padded_images = []
    for img in images:
        if img.shape[0] < max_channels:
            padding = torch.zeros(
                max_channels - img.shape[0], img.shape[1], img.shape[2]
            )
            img_padded = torch.cat([img, padding], dim=0)
        elif img.shape[0] > max_channels:
            img_padded = img[:max_channels]
        else:
            img_padded = img
        padded_images.append(img_padded)

    # Stack into batch
    batched_images = torch.stack(padded_images, dim=0)

    if has_labels:
        # Handle different label types
        if isinstance(labels[0], torch.Tensor):
            batched_labels = torch.stack(labels, dim=0)
        else:
            batched_labels = torch.tensor(labels)
        return batched_images, batched_labels

    return batched_images


def train_vae(
    vae: UniversalVAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    lr: float = 1e-4,
    beta_start: float = 1.0,
    beta_end: float = 4.0,
    device: torch.device = torch.device("cpu"),
    save_path: str = "vae_checkpoint.pth",
    patience: int = 10,
):
    """
    Train the Universal VAE.

    Args:
        vae: VAE model
        train_loader: Training data
        val_loader: Validation data (optional)
        epochs: Number of training epochs
        lr: Learning rate
        beta_start: Start KL divergence weight (beta-VAE)
        beta_end: Final KL divergence weight
        device: 'cuda', 'mps' or 'cpu'
        save_path: Where to save the best model
        patience: Number of epochs with no improvement after which training will be stopped
    """
    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    patience_counter = 0

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "train_kl": [],
        "beta_values": [],
        "learning_rates": [],
    }
    # train_losses = []
    # val_losses = []

    for epoch in range(epochs):
        # Training
        vae.train()
        # train_loss = 0
        best_train_loss = float("inf")
        train_metrics = {"total": 0, "recon": 0, "kl": 0}
        # Linear Beta-Schedule
        beta = (
            beta_start + (beta_end - beta_start) * (epoch / (epochs - 1))
            if epochs > 1
            else beta_end
        )

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch

            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data)
            losses = vae_loss(recon_batch, data, mu, logvar, beta)

            losses["total"].backward()
            # train_loss += losses.item()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = data.size(0)
            train_metrics["total"] += losses["total"].item() / batch_size
            train_metrics["recon"] += losses["recon"].item() / batch_size
            train_metrics["kl"] += losses["kl"].item() / batch_size

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {losses['total'].item() / batch_size:.4f}"
                    f"(β={beta:.2f})"
                )

        # Average training metrics
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        history["train_loss"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # avg_train_loss = train_loss / len(train_loader.dataset)
        # train_losses.append(avg_train_loss)

        # Validation
        if val_loader is not None:
            vae.eval()
            val_metrics = {"total": 0.0, "recon": 0.0, "kl": 0.0}

            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        data = batch[0]
                    else:
                        data = batch

                    data = data.to(device)
                    recon_batch, mu, logvar = vae(data)
                    val_losses = vae_loss(recon_batch, data, mu, logvar, beta).item()

                    batch_size = data.size(0)
                    val_metrics["total"] += val_losses["total"].item() / batch_size
                    val_metrics["recon"] += val_losses["recon"].item() / batch_size
                    val_metrics["kl"] += val_losses["kl"].item() / batch_size

            # Average validation metrics
            for key in val_metrics:
                val_metrics[key] /= len(val_loader)

            history["val_loss"].append(val_metrics["total"])

            print(f"\nEpoch {epoch + 1} Summary:")
            print(
                f"  Train Loss: {train_metrics['total']:.4f} "
                f"(Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f})"
            )
            print(
                f"  Val Loss:   {val_metrics['total']:.4f} "
                f"(Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f})"
            )
            print(f"  β: {beta:.2f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 60)

            # Early stopping and model saving
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": vae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss,
                        "beta": beta,
                        "history": history,
                    },
                    save_path,
                )
                print(f"Saved best model (val_loss: {best_val_loss:.4f})\n")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                    break

            # avg_val_loss = val_loss / len(val_loader.dataset)
            # val_losses.append(avg_val_loss)
            #
            #
            # print(f'Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, '
            #       f'Val Loss: {avg_val_loss:.4f}')
            #
            # scheduler.step(avg_val_loss)
            #
            # # Save best model
            # if avg_val_loss < best_val_loss:
            #     best_val_loss = avg_val_loss
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': vae.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': best_val_loss,
            #     }, save_path)
            #     print(f'Saved best model to {save_path}')
        else:
            csv_file = "res_vae_results.csv"
            if not Path(csv_file).exists():
                with open(csv_file, "w") as f:
                    f.write("epoch,train_loss\n")
            with open(csv_file, "a") as f:
                f.write(f"{epoch},{history['train_loss'][epoch]}\n")

            if train_metrics["total"] < best_train_loss:
                best_loss = train_metrics["total"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": vae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                    },
                    save_path,
                )
                print(f"Saved best model to {save_path}")

            print(
                f"\nEpoch {epoch + 1}: Train Loss = {train_metrics['total']:.4f} "
                f"(β={beta:.2f})"
            )
            print("-" * 60)

        # Update learning rate
        scheduler.step()

    return history


def plot_history(history: dict, save_path: str = "training_history.png"):
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Training History")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["train_recon"], label="Reconstruction")
    axes[0, 1].plot(history["train_kl"], label="KL Divergence")
    axes[0, 1].set_title("Loss Components")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["beta_values"])
    axes[1, 0].set_title("β Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("β Value")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["learning_rates"])
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def visualize_reconstructions(
    loader,
    vae_model,
    device,
    n_samples: int = 6,
    save_path: str = "reconstructions.png",
):
    vae_model.to(device)
    vae_model.eval()
    # Grab one batch
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        data = batch[0]
    else:
        data = batch
    data = data[:n_samples].to(device)

    with torch.no_grad():
        recon, _, _ = vae_model(data)
        # If outputs are logits, squash them; otherwise clamp to [0,1]
        try:
            recon = torch.sigmoid(recon)
        except Exception:
            recon = torch.clamp(recon, 0.0, 1.0)

    data = data.cpu()
    recon = recon.cpu()

    n = data.size(0)
    cols = n
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2, 4))
    for i in range(n):
        orig = data[i]
        rec = recon[i]

        # Choose up to 3 channels for visualization (RGB). If single-channel, repeat.
        def to_hwc(t):
            c = t.shape[0]
            if c >= 3:
                img = t[:3]
            elif c == 1:
                img = t.repeat(3, 1, 1)
            else:
                # If 2+ channels but <3, pad with zeros
                pad = torch.zeros(3 - c, t.shape[1], t.shape[2])
                img = torch.cat([t, pad], dim=0)
            img = img.permute(1, 2, 0).numpy()
            # Clip for safety
            img = img.clip(0.0, 1.0)
            return img

        axes[0, i].imshow(to_hwc(orig))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(to_hwc(rec))
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    MAX_CHANNELS = 16
    LATENT_DIM = 32
    IMAGE_SIZE = 128
    BATCH_SIZE = 64
    EPOCHS = 50
    BETA_START = 1.0
    BETA_END = 1.0

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using MPS")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU")

    # Load the datasets
    print("\n=== Loading Datasets ===")
    dataset_names = [
        "mnist",
        "fashion",
        "cifar10",
        "stl10",
        "cxr8",
        "brain_tumor",
        "eurosat_ms",
    ]
    datasets = []
    for d in dataset_names:
        dm = DataManager(
            cfg=None, batch_size=BATCH_SIZE, seed=1, pixel_size=IMAGE_SIZE, dataset=d
        )
        datasets.append(
            MultiModalImageDataset(
                dm.get_dataset(num_points=9000),
                MAX_CHANNELS,
                IMAGE_SIZE,
                dataset_name=d,
                preserve_channels=True,
            )
        )

    # Combine all datasets for universal VAE training
    combined_dataset = ConcatDataset(datasets)

    # CRITICAL: Use custom collate function to handle variable channels
    collate_fn = partial(custom_collate_fn, max_channels=MAX_CHANNELS)

    train_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Initialize VAE
    vae = UniversalVAE(
        max_channels=MAX_CHANNELS, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE
    )

    print(f"Model parameters: {sum(p.numel() for p in vae.parameters()):,}")

    # Train VAE
    # print("\n=== Training Universal VAE ===")
    # train_history = train_vae(
    #     vae, train_loader,
    #     epochs=EPOCHS,
    #     lr=1e-4,
    #     beta_start=BETA_START,
    #     beta_end=BETA_END,
    #     device=DEVICE,
    #     save_path='universal_res_vae_5.pth',
    #     patience=10
    # )

    # Load trained VAE
    # Load trained model (for demonstration, assume pre-trained)
    load_model_path = "results1/universal_vae.pth"
    if os.path.exists(load_model_path):
        checkpoint = torch.load(load_model_path, map_location=DEVICE)
        vae.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded pre-trained VAE model.")

    # train_history_path = 'res.png'
    # plot_history(train_history, save_path=train_history_path)

    # Extract fingerprints for each dataset
    print("\n=== Extracting Dataset Fingerprints ===")
    fingerprinter = DatasetFingerprinter(vae, device=DEVICE)

    fingerprints = {}
    for ds, name in zip(datasets, dataset_names):
        # Use same collate function for individual datasets
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )
        visualize_reconstructions(
            loader, vae, DEVICE, save_path=f"reconstructions_{name}.png"
        )
        fp = fingerprinter.extract_fingerprint(loader, name)
        fingerprints[name] = fp

        print(f"\n{name} Fingerprint:")
        print(f"  Mean latent: {fp['mean'][:5]}... (first 5 dims)")
        print(f"  Std latent: {fp['std'][:5]}...")
        print(f"  Samples: {fp['n_samples']}")

    # Compare datasets
    results = []
    print("\n=== Dataset Comparisons ===")
    for name1 in fingerprints:
        for name2 in fingerprints:
            if name1 < name2:  # Avoid duplicates
                distances = fingerprinter.compare_fingerprints(
                    fingerprints[name1], fingerprints[name2]
                )
                results.append({"dataset1": name1, "dataset2": name2, **distances})
                print(f"\n{name1} vs {name2}:")
                for metric, value in distances.items():
                    print(f"  {metric}: {value:.4f}")

    # Visualize
    print("\n=== Generating Visualizations ===")
    visualisations_path = "results1/dataset_fingerprints_vae.png"
    fingerprinter.visualize_fingerprints(fingerprints, save_path=visualisations_path)

    # Save fingerprints
    fingerprints_path = "results1/dataset_fingerprints_vae.pkl"
    with open(fingerprints_path, "wb") as f:
        pickle.dump(fingerprints, f)
    print("\nSaved fingerprints to dataset_fingerprints.pkl")

    comparison_results_path = "results1/dataset_comparison_results_vae.pkl"
    with open(comparison_results_path, "wb") as f:
        pickle.dump(results, f)

    print("\nTraining complete!")
    print(f"Saved results to: {comparison_results_path}")
    print(f"Saved fingerprints to: {fingerprints_path}")
    # print(f"Saved visualizations to: {train_history_path}, {visualisations_path}")
