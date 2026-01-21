from typing import Any, Dict, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from vae import UniversalVAE


class DatasetFingerprinter:
    """Enhanced fingerprint extraction with more metrics"""

    def __init__(self, vae: UniversalVAE, device: torch.device = torch.device("cpu")):
        self.vae = vae
        self.device = device
        self.vae.to(device)
        self.vae.eval()

    def extract_fingerprint(
        self, dataloader: DataLoader, dataset_name: str, n_samples: int = 1000
    ) -> Dict[str, Any]:
        """Extract comprehensive fingerprint with multiple statistics"""
        all_latents = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                latents = self.vae.get_latent(images, sample=False)
                all_latents.append(latents.cpu().numpy())

                # Stop if we have enough samples
                total = sum([len(l) for l in all_latents])
                if total >= n_samples:
                    break

        all_latents = np.vstack(all_latents)[:n_samples]  # Limit to n_samples

        # Compute comprehensive statistics
        fingerprint = {
            "name": dataset_name,
            "mean": np.mean(all_latents, axis=0),
            "std": np.std(all_latents, axis=0),
            "cov": np.cov(all_latents.T),
            "latents": all_latents,
            "n_samples": len(all_latents),
        }

        return fingerprint

    def compare_fingerprints(self, fp1: Dict, fp2: Dict) -> Dict[str, float]:
        """Compare two fingerprints using multiple metrics"""
        metrics = {}

        # Basic distances
        metrics["mean_euclidean"] = np.linalg.norm(fp1["mean"] - fp2["mean"])
        metrics["mean_cosine"] = 1 - np.dot(fp1["mean"], fp2["mean"]) / (
            np.linalg.norm(fp1["mean"]) * np.linalg.norm(fp2["mean"]) + 1e-8
        )

        # Mahalanobis distance
        try:
            pooled_cov = (fp1["cov"] + fp2["cov"]) / 2
            inv_cov = np.linalg.inv(pooled_cov + 1e-6 * np.eye(pooled_cov.shape[0]))
            diff = fp1["mean"] - fp2["mean"]
            metrics["mahalanobis"] = np.sqrt(diff.T @ inv_cov @ diff)
        except:
            metrics["mahalanobis"] = np.nan

        # Covariance similarity
        metrics["cov_frobenius"] = np.linalg.norm(fp1["cov"] - fp2["cov"], "fro")

        return metrics

    def visualize_fingerprints(
        self, fingerprints: Dict[str, Dict], save_path: Optional[str] = None
    ):
        """Visualisation of the fingerprints"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. t-SNE plot
        self._plot_tsne(axes[0, 0], fingerprints)

        # 2. Mean vectors (PCA)
        self._plot_mean_vectors(axes[0, 1], fingerprints)

        # 3. Standard deviations
        self._plot_std_devs(axes[0, 2], fingerprints)

        # 4. Covariance heatmap (averaged)
        self._plot_covariance(axes[1, 0], fingerprints)

        # 5. Distance matrix
        self._plot_distance_matrix(axes[1, 2], fingerprints)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        plt.show()

    def _plot_tsne(self, ax, fingerprints):
        """Plot t-SNE of latent codes"""
        all_latents = []
        all_labels = []

        for name, fp in fingerprints.items():
            latents = fp["latents"][:500]  # Subsample for speed
            all_latents.append(latents)
            all_labels.extend([name] * len(latents))

        all_latents = np.vstack(all_latents)

        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latents_2d = tsne.fit_transform(all_latents)

        for name in fingerprints.keys():
            mask = np.array([l == name for l in all_labels])
            ax.scatter(
                latents_2d[mask, 0], latents_2d[mask, 1], label=name, alpha=0.6, s=10
            )
        ax.set_title("t-SNE of Latent Codes")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_mean_vectors(self, ax, fingerprints):
        """Plot PCA of mean vectors"""
        means = np.array([fp["mean"] for fp in fingerprints.values()])
        names = list(fingerprints.keys())

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        means_2d = pca.fit_transform(means)

        ax.scatter(means_2d[:, 0], means_2d[:, 1], s=100, alpha=0.7)
        for i, name in enumerate(names):
            ax.annotate(name, (means_2d[i, 0], means_2d[i, 1]), fontsize=9, ha="center")
        ax.set_title("Dataset Means (PCA)")
        ax.grid(True, alpha=0.3)

    def _plot_std_devs(self, ax, fingerprints):
        """Plot average standard deviations"""
        names = list(fingerprints.keys())
        mean_stds = [np.mean(fp["std"]) for fp in fingerprints.values()]

        bars = ax.bar(names, mean_stds)
        ax.set_title("Average Latent Standard Deviation")
        ax.set_ylabel("Mean σ")
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    def _plot_covariance(self, ax, fingerprints):
        """Plot average covariance matrix"""
        mean_cov = np.mean([fp["cov"] for fp in fingerprints.values()], axis=0)

        im = ax.imshow(mean_cov, cmap="RdBu_r", aspect="auto")
        ax.set_title("Mean Covariance Matrix")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Latent Dimension")
        plt.colorbar(im, ax=ax)

    def _plot_distance_matrix(self, ax, fingerprints):
        """Plot distance matrix between datasets"""
        names = list(fingerprints.keys())
        n = len(names)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    dists = self.compare_fingerprints(
                        fingerprints[names[i]], fingerprints[names[j]]
                    )
                    distance_matrix[i, j] = dists.get("mahalanobis", np.nan)

        im = ax.imshow(distance_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title("Dataset Distance Matrix (Mahalanobis)")

        # Add text annotations
        for i in range(n):
            for j in range(n):
                if i != j:
                    text = ax.text(
                        j,
                        i,
                        f"{distance_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )


# class DatasetFingerprinterOld:
#     """
#     Extracts latent-space dataset fingerprints using trained VAE.
#     """
#
#     def __init__(self, vae: UniversalVAE, device: str = 'cuda'):
#         self.vae = vae
#         self.device = device
#         self.vae.to(device)
#         self.vae.eval()
#
#     def extract_fingerprint(self,
#                             dataloader: DataLoader,
#                             dataset_name: str) -> Dict[str, np.ndarray]:
#         """
#         Extract (μ_z, Σ_z) fingerprint from dataset.
#
#         Args:
#             dataloader: DataLoader for the dataset
#             dataset_name: Name for tracking
#
#         Returns:
#             Dictionary with mean, cov, and latent codes
#         """
#         all_latents = []
#
#         with torch.no_grad():
#             for batch in dataloader:
#                 if isinstance(batch, (list, tuple)):
#                     images = batch[0]
#                 else:
#                     images = batch
#
#                 images = images.to(self.device)
#                 latents = self.vae.get_latent(images, sample=False)
#                 all_latents.append(latents.cpu().numpy())
#
#         # Stack all latent codes
#         all_latents = np.vstack(all_latents)
#
#         # Compute statistics
#         fingerprint = {
#             'name': dataset_name,
#             'mean': np.mean(all_latents, axis=0),  # μ_z ∈ R^latent_dim
#             'std': np.std(all_latents, axis=0),
#             'cov': np.cov(all_latents.T),  # Σ_z ∈ R^(latent_dim × latent_dim)
#             'latents': all_latents,  # Keep for visualization
#             'n_samples': len(all_latents)
#         }
#
#         return fingerprint
#
#     def compare_fingerprints(self,
#                              fp1: Dict,
#                              fp2: Dict) -> Dict[str, float]:
#         """
#         Compare two dataset fingerprints.
#
#         Returns:
#             Dictionary of distance metrics
#         """
#         # Euclidean distance between means
#         mean_dist = np.linalg.norm(fp1['mean'] - fp2['mean'])
#
#         # Mahalanobis distance (using pooled covariance)
#         pooled_cov = (fp1['cov'] + fp2['cov']) / 2
#         try:
#             inv_cov = np.linalg.inv(pooled_cov + 1e-6 * np.eye(pooled_cov.shape[0]))
#             diff = fp1['mean'] - fp2['mean']
#             mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
#         except:
#             mahal_dist = np.nan
#
#         # Frobenius norm of covariance difference
#         cov_dist = np.linalg.norm(fp1['cov'] - fp2['cov'], ord='fro')
#
#         return {
#             'mean_euclidean': float(mean_dist),
#             'mahalanobis': float(mahal_dist),
#             'covariance_frobenius': float(cov_dist)
#         }
#
#     def visualize_fingerprints(self,
#                                fingerprints: Dict[str, Dict],
#                                save_path: Optional[str] = None):
#         """
#         Visualize dataset fingerprints using t-SNE and covariance heatmaps.
#         """
#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#
#         # Left: t-SNE of all latent codes
#         all_latents = []
#         all_labels = []
#
#         for name, fp in fingerprints.items():
#             all_latents.append(fp['latents'])
#             all_labels.extend([name] * len(fp['latents']))
#
#         all_latents = np.vstack(all_latents)
#
#         # Subsample if too large
#         if len(all_latents) > 5000:
#             idx = np.random.choice(len(all_latents), 5000, replace=False)
#             all_latents = all_latents[idx]
#             all_labels = [all_labels[i] for i in idx]
#
#         print("Computing t-SNE...")
#         tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#         latents_2d = tsne.fit_transform(all_latents)
#
#         # Plot t-SNE
#         for name in fingerprints.keys():
#             mask = np.array([l == name for l in all_labels])
#             axes[0].scatter(latents_2d[mask, 0], latents_2d[mask, 1],
#                             label=name, alpha=0.6, s=10)
#         axes[0].set_title('t-SNE of Latent Codes (colored by dataset)')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3)
#
#         # Right: Mean covariance heatmap
#         mean_cov = np.mean([fp['cov'] for fp in fingerprints.values()], axis=0)
#         im = axes[1].imshow(mean_cov, cmap='RdBu_r', aspect='auto')
#         axes[1].set_title('Mean Latent Covariance Matrix')
#         axes[1].set_xlabel('Latent Dimension')
#         axes[1].set_ylabel('Latent Dimension')
#         plt.colorbar(im, ax=axes[1])
#
#         plt.tight_layout()
#
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"Saved visualization to {save_path}")
#
#         plt.show()
