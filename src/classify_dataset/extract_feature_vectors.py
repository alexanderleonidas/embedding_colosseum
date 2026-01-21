import ast
import csv
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import seaborn as sns
import torch
import tqdm
from edge_density import edge_density
from glcm import glcm
from matplotlib import pyplot as plt
from phase_correlation import compute_symmetry_score
from shannon_entropy import shannon_entropy
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset.DataManager import DataManager


def extract_features_vectors(dataloader: DataLoader, normalise: bool = True) -> dict:
    """
    Compute the feature vectors for all images in a PyTorch DataLoader.

    Args:
        dataloader: PyTorch DataLoader yielding (images, labels) or images
        normalise: Whether to normalise each metrics to [0,1]

    Returns:
        Dict containing

            - ``feature vectors``: Dict of feature vectors for each image
            - ``mean_vector``: Mean entropy across dataset
            - ``std_entropy``: Standard deviation of entropy
            - ``min_entropy``: Minimum entropy
            - ``max_entropy``: Maximum entropy
            - ``per_class_stats``: Dictionary with entropy stats per class (if labels available)
    """

    feature_vectors = []
    all_labels = []
    class_specific_feature_vectors = defaultdict(list)

    print(f"Computing Feature Vectors...")
    # Process batches
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(
            tqdm.tqdm(dataloader, desc="Processing images")
        ):
            # Handle different DataLoader return formats
            images, labels = batch_data[0], batch_data[1]

            # Process each image in the batch
            for i in range(images.shape[0]):
                # global_index = batch_idx * images.shape[0] + i
                image = images[i]

                # normalise image to [0, 1] if needed
                if image.max() > 1.0:
                    image = (image - image.min()) / (image.max() - image.min() + 1e-10)

                # convert to grayscale if needed
                if image.shape[0] > 1:
                    # Convert to grayscale if image has multiple channels
                    image = torch.mean(image, dim=0, keepdim=True)

                # Calculate features
                entropy = shannon_entropy(image, normalise=normalise)
                density = edge_density(image, threshold=0.1, normalise=normalise)
                phase_corr = compute_symmetry_score(image)
                glcm_stats = glcm(image)

                # Append feature to class specific list
                feature_vector = np.array(
                    [
                        entropy,
                        density,
                        phase_corr,
                        glcm_stats["homogeneity"],
                        glcm_stats["correlation"],
                        glcm_stats["energy"],
                    ]
                )
                feature_vectors.append(feature_vector)

                # Store label if available
                if labels is not None:
                    label = (
                        labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                    )
                    all_labels.append(label)
                    class_specific_feature_vectors[label].append(feature_vector)

    # Compute overall statistics
    if feature_vectors:
        f_vec = np.vstack(feature_vectors)
        results = {
            "feature_vectors": feature_vectors,
            "mean_vector": f_vec.mean(axis=0).tolist(),
            "std_vector": f_vec.std(axis=0).tolist(),
            "covariance": np.cov(f_vec, rowvar=False).tolist(),
        }

    if class_specific_feature_vectors:
        per_class_stats = {}
        for c, f in class_specific_feature_vectors.items():
            fc_vec = np.vstack(f)
            per_class_stats[c] = {
                "features_vectors": fc_vec.tolist(),
                "mean_vector": fc_vec.mean(axis=0).tolist(),
                "std_vector": fc_vec.std(axis=0).tolist(),
                "covariance": np.cov(fc_vec, rowvar=False).tolist(),
                "count": len(feature_vectors),
            }
        results["per_class_stats"] = per_class_stats

    return results


def save_dataset_fingerprint_results_to_csv(
    results: dict, dataset: str, pixel_size: int, file_name: str
):
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode="a", newline="") as csvfile:
        fieldnames = [
            "dataset",
            "pixel_size",
            "mean_vector",
            "std_vector",
            "covariance",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "dataset": dataset,
                "pixel_size": pixel_size,
                "mean_vector": results.get("mean_vector"),
                "std_vector": results.get("std_vector"),
                "covariance": results.get("covariance"),
            }
        )


def save_class_specific_results_npz(results: Dict, file_name: str) -> str:
    """
    Save `results['per_class_stats']` into a compressed .npz file.

    - Each class produces keys: class_<label>_features, class_<label>_mean, class_<label>_std,
      class_<label>_covariance, class_<label>_count
    - A top-level 'classes' array lists the class labels (as strings).
    Returns the `file_path`.
    """
    if not isinstance(results, dict) or "per_class_stats" not in results:
        raise ValueError("`results` must be a dict containing a 'per_class_stats' key")

    per_class = results["per_class_stats"]
    if not per_class:
        raise ValueError("`per_class_stats` is empty")

    with open(file_name, mode="wb") as f:
        pickle.dump(per_class, f)

    # with open(file_name, mode='rb') as f:
    #     loaded_data = pickle.load(f)
    #     print(f"Loaded data from {file_name}")
    #     for class_label, stats in loaded_data.items():
    #         print(f"Class {class_label}:")
    #         print(f"  Count: {stats['count']}")
    #         print(f"  Mean Vector: {stats['mean_vector']}")
    #         print(f"  Std Vector: {stats['std_vector']}")
    #         print(f"  Covariance: {stats['covariance']}")


def mahalanobis_distance_between_means(
    mean1, cov1, mean2, cov2, regularization: float = 1e-6, use_pooled: bool = True
) -> float:
    m1 = np.asarray(mean1, dtype=float)
    m2 = np.asarray(mean2, dtype=float)
    c1 = np.asarray(cov1, dtype=float)
    c2 = np.asarray(cov2, dtype=float)

    if m1.shape != m2.shape:
        raise ValueError("Mean vectors must have the same shape")

    cov = (c1 + c2) / 2.0 if use_pooled else c1
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance must be a square matrix")

    cov = cov + np.eye(cov.shape[0]) * regularization
    inv_cov = np.linalg.pinv(cov)
    diff = m1 - m2
    dist_sq = float(diff.T @ inv_cov @ diff)
    return float(np.sqrt(max(dist_sq, 0.0)))


def compute_pairwise_mahalanobis(
    results_map: Dict[str, dict], regularization: float = 1e-6, use_pooled: bool = True
):
    names = list(results_map.keys())
    n = len(names)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            ra = results_map[names[i]]
            rb = results_map[names[j]]
            dist = mahalanobis_distance_between_means(
                ra["mean_vector"],
                ra["covariance"],
                rb["mean_vector"],
                rb["covariance"],
                regularization=regularization,
                use_pooled=use_pooled,
            )
            mat[i, j] = mat[j, i] = dist
    return names, mat


def plot_mahalanobis_heatmap(
    names, matrix, out_file: str = None, figsize=(8, 6), cmap="YlOrRd"
):
    plt.figure(figsize=figsize)
    # sns.set(style="white")
    ax = sns.heatmap(
        matrix,
        xticklabels=names,
        yticklabels=names,
        cmap=cmap,
        annot=True,
        fmt=".3f",
        square=True,
    )
    plt.title("Pairwise Mahalanobis Distance Heatmap")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=200)
    plt.show()


def get_feature_vectors(
    dataset_results: Dict[str, dict], num_vectors_per_dataset: int
) -> tuple[np.ndarray, list]:
    """
    Extract a specified number of feature vectors from all datasets in dataset_results.

    Args:
        dataset_results: Dict with dataset names as keys and results dicts as values
        num_vectors_per_dataset: Number of vectors to take from each dataset

    Returns:
        Tuple of (array of feature vectors, list of corresponding dataset labels)
    """
    vectors = []
    labels = []
    for dataset_name, results in dataset_results.items():
        dataset_vectors = []
        dataset_labels = []
        for class_label, stats in results["per_class_stats"].items():
            dataset_vectors.extend(np.array(stats["features_vectors"]))
            dataset_labels.extend([dataset_name] * len(stats["features_vectors"]))

        indexs = np.random.choice(
            np.arange(0, len(dataset_vectors) - 1),
            num_vectors_per_dataset,
            replace=False,
        )
        for i in indexs:
            vectors.append(dataset_vectors[i])
            labels.append(dataset_labels[i])

    return np.array(vectors), labels


def plot_tsne(
    feature_vectors, labels=None, out_file: str = None, figsize=(8, 6), perplexity=30
):
    """
    Plot TSNE visualization of feature vectors.

    Args:
        feature_vectors: List of feature vectors
        labels: Optional list of labels for coloring
        out_file: Optional file path to save the plot
        figsize: Figure size
        perplexity: TSNE perplexity parameter
    """
    f_vec = np.vstack(feature_vectors)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(f_vec)

    plt.figure(figsize=figsize)
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                color=colors[i],
                label=str(label),
                alpha=0.7,
            )
        plt.legend()
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
    plt.title("TSNE Plot of Feature Vectors")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=200)
    plt.show()


def load_results(run_id):
    """
    Load dataset fingerprint results from CSV and class-specific results from pickle files.
    """
    dataset_fingerprints = {}
    with open(f"{run_id}_dataset_fingerprint_results.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_fingerprints[row["dataset"]] = {
                "mean_vector": ast.literal_eval(row["mean_vector"]),
                "std_vector": ast.literal_eval(row["std_vector"]),
                "covariance": ast.literal_eval(row["covariance"]),
            }
    for d in datasets:
        dataset_fingerprints[d]["per_class_stats"] = pickle.load(
            open(f"{run_id}_class_specific_feature_vectors_results_{d}_{128}.pkl", "rb")
        )

    return dataset_fingerprints


# Example usage
if __name__ == "__main__":
    run_id = time.strftime("%Y%m%d-%H%M%S")
    datasets = [
        "mnist",
        "fashion",
        "cifar10",
        "stl10",
        "cxr8",
        "brain_tumor",
        "eurosat_rgb",
    ]
    pixel_sizes = [128]
    dataset_results = {}
    # for d in datasets:
    #     for p in pixel_sizes:
    #         dm = DataManager(cfg=None, batch_size=100, seed=42, pixel_size=p, dataset=d)
    #         train, _, _ = dm.get_loaders(1.0, 0, 0)
    #         print(f"Extracting {len(train.dataset)} feature vectors for Dataset: {d}, Pixel Size: {p}, Run ID: {run_id}, ")
    #
    #         results = extract_features_vectors(train, normalise=True)
    #         save_dataset_fingerprint_results_to_csv(results, d, p, file_name=f"{run_id}_dataset_fingerprint_results.csv")
    #         save_class_specific_results_npz(results,  file_name=f"{run_id}_class_specific_feature_vectors_results_{d}_{p}.pkl")
    #         dataset_results[d] = results
    #
    #         # print(f"Feature vectors for {d} dataset: {results['feature_vectors']}")
    #         print(f"Feature extraction of {len(results['feature_vectors'])} images complete for {d}.")
    #         print(f"Mean Feature Vector: {results['mean_vector']}")
    #         print(f"Std Feature Vector: {results['std_vector']}")
    #         print(f"Covariance: {results['covariance']}")

    dataset_fingerprints = load_results("hand_picked_results/20260119-174435")
    names, mat = compute_pairwise_mahalanobis(dataset_fingerprints)
    plot_mahalanobis_heatmap(names, mat, out_file=f"{run_id}_mahalanobis_heatmap.png")
    random_vectors, random_labels = get_feature_vectors(dataset_fingerprints, 1000)
    plot_tsne(random_vectors, out_file=f"{run_id}_tsne.png", labels=random_labels)
