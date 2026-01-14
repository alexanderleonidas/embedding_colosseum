import logging
import time

import pennylane as qml
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pennylane import numpy as np
from rich import print, progress
from tqdm import tqdm

from src.dataset.DataManager import DataManager
from src.embeddings.AngleEncoding import AngleEncodingEmbedding
from src.embeddings.FRQI_PennyLane import FRQI
from src.embeddings.NEQR_PennyLane import NEQR
from src.embeddings.OQIM_PennyLane import OQIM
from src.embeddings.NAQSS import NAQSS
from src.embeddings.RMP_Prototype import RMPEmbedding
from src.embeddings.ZZFeatureMap import ZZFeatureMapEmbedding
from src.model.VariationalClassifier import VariationalClassifier
from src.preprocessing.pca import transform_to_pca_loader
from src.utils.save_training_progress import TrainingLogger

log = logging.getLogger(__name__)
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


@torch.no_grad()  # Ensure no gradients are computed during accuracy calculation
def accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    num_correct = (labels == predictions).sum().item()
    return num_correct / len(labels)


def run_classifier(cfg):
    training_logger = TrainingLogger(run_id=int(time.time()), cfg=cfg)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    log.info(f"Using {device}.")

    # Set up csv for training results
    csv_fp = HydraConfig.get().runtime.output_dir + "/results.csv"
    with open(csv_fp, "w") as f:
        f.write("epoch,batch,train_loss,train_acc,val_loss,val_acc\n")

    # "pixel size" Can be changed in /conf/training/base.yaml
    # embedding in /conf/config.yaml
    if cfg.embedding == "FRQI":
        embedding = FRQI(num_pixels=cfg.training.image_width * cfg.training.image_width)
    elif cfg.embedding == "NEQR":
        embedding = NEQR(num_pixels=cfg.training.image_width * cfg.training.image_width)
    elif cfg.embedding == "ZZFeatureMap":
        embedding = ZZFeatureMapEmbedding(num_features=6)
    elif cfg.embedding == "AngleEncoding":
        embedding = AngleEncodingEmbedding(num_features=6)
    elif cfg.embedding == "RMP":
        embedding = RMPEmbedding(num_features=6, alpha=0.5)
    elif cfg.embedding == "OQIM":
        embedding = OQIM(num_pixels=cfg.training.image_width * cfg.training.image_width)
    elif cfg.embedding == "NAQSS":
        embedding = NAQSS(num_pixels=cfg.training.image_width * cfg.training.image_width)
    else:
        raise ValueError("Unknown embedding method")

    dm = DataManager(
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
        dataset=cfg.dataset.name,
        pixel_size=cfg.training.image_width,  # pixel size set above
        transform=cfg.image_preprocessing,
        make_binary=cfg.dataset.binary,
    )
    train_loader, validation_loader, test_loader = dm.get_loaders(
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
    )

    # applying PCA for PQC-style embeddings
    if cfg.embedding in ["ZZFeatureMap", "AngleEncoding", "RMP"]:
        train_loader, validation_loader, test_loader = transform_to_pca_loader(
            train_loader,
            validation_loader,
            test_loader,
            batch_size=cfg.training.batch_size,
            n_components=embedding.num_features,
        )

    model = VariationalClassifier(
        num_qubits=embedding.num_qubits,
        num_layers=cfg.model.num_layers,
        num_classes=cfg.dataset.num_classes,
        num_pixels=cfg.training.image_width * cfg.training.image_width,
        state_preparation=embedding.state_preparation,
    )
    log.info(f"Weights Shape: {model.weights.shape}")
    log.info(f"Bias Shape: {model.bias.shape}")

    optimizer = torch.optim.AdamW(
        params=[model.weights, model.bias],
        lr=cfg.training.lr,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training Loop
    total_samples = len(train_loader.dataset)
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        with progress.Progress() as p:
            task = p.add_task(
                f"[yellow]Training Epoch {epoch + 1}/{cfg.training.epochs}..."
                + f"(Img: {cfg.training.image_width}, Emb: {cfg.embedding})",
                total=len(train_loader),
            )
            for batch, (X, y) in enumerate(train_loader):
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                if cfg.embedding in ["NEQR", "OQIM", "NAQSS"]:
                    # Batch processing is not supported by these embeddings
                    pred = model.classify(X, batch_processing=False)
                else:
                    pred = model.classify(X)

                loss = loss_fn(pred, y.long())

                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                current_cost = loss.item()
                epoch_loss += current_cost

                # Compute accuracy
                acc = accuracy(logits=pred, labels=y)
                epoch_acc += acc
                current = batch * cfg.training.batch_size

                log.info(
                    f"[{current:>5d}/{total_samples:>5d}] | "
                    + f"Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}"
                )

                # Save training results per batch
                with open(csv_fp, "a") as f:
                    f.write(f"{epoch},{batch},{current_cost},{acc},,\n")

                p.update(task, advance=1)

        # Accumulate epoch metrics
        epoch_loss /= total_samples
        epoch_acc /= total_samples

        # Validation loop
        with torch.no_grad():
            val_acc, val_loss = 0.0, 0.0
            with progress.Progress() as p:
                task = p.add_task(
                    f"[yellow]Validation Epoch {epoch + 1}/{cfg.training.epochs}..."
                    + f"(Img: {cfg.training.image_width}, Emb: {cfg.embedding})",
                    total=len(validation_loader),
                )
                for X, y in validation_loader:
                    X = X.to(device)
                    y = y.to(device)

                    # batch processing not supported by these embeddings, conditional also for validation loop
                    if cfg.embedding in ["NEQR", "OQIM", "NAQSS"]: 
                        pred = model.classify(X, batch_processing=False)
                    else:
                        pred = model.classify(X)

                    val_loss += loss_fn(pred, y.long()).item()
                    val_acc += accuracy(logits=pred, labels=y)
                    p.update(task, advance=1)

        val_loss /= len(validation_loader)
        val_acc /= len(validation_loader)
        log.info(
            f"Validation for epoch {epoch:>3} | Cost: {val_loss:0.7f} | Accuracy: {val_acc:0.7f}"
        )
        with open(csv_fp, "a") as f:
            f.write(f"{epoch},{batch},,,{val_loss},{val_acc}\n")

        training_logger.save_model_weights_and_bias(model.weights, model.bias, epoch)
    return val_loss
