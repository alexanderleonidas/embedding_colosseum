import logging
import time

import numpy as np
import torch
from hydra.utils import instantiate
from rich import print, progress

from src.dataset.DataManager import DataManager
from src.embeddings.AngleEncoding import AngleEncodingEmbedding
from src.embeddings.FRQI_PennyLane import FRQI
from src.embeddings.NAQSS import NAQSS
from src.embeddings.NEQR_PennyLane import NEQR
from src.embeddings.OQIM_PennyLane import OQIM
from src.embeddings.RMP_Prototype import RMPEmbedding
from src.embeddings.ZZFeatureMap import ZZFeatureMapEmbedding
from src.model.VariationalClassifier import VariationalClassifier
from src.preprocessing.pca import transform_to_pca_loader
from src.utils.TrainingLogger import TrainingLogger

log = logging.getLogger(__name__)
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def run_classifier(cfg):
    log.info(f"Using {device}.")
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)

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
        embedding = NAQSS(
            num_pixels=cfg.training.image_width * cfg.training.image_width
        )
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
        test_split=0.1,  # Test set not used as no hyperparameter tuning is performed
    )
    # Dataset Info
    log.info(
        f"Dataset: {cfg.dataset.name} | "
        + f"Train Size: {len(train_loader.dataset)} | "
        + f"Validation Size: {len(validation_loader.dataset)} | "
        + f"Test Size: {len(test_loader.dataset)}"
    )
    training_logger = TrainingLogger(
        cfg, train_loader=train_loader, val_loader=validation_loader
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
    for epoch in range(cfg.training.epochs):
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

                # Compute accuracy
                training_logger.log_training(
                    pred=pred,
                    labels=y,
                    loss=current_cost,
                    epoch=epoch,
                    batch=batch,
                )
                p.update(task, advance=1)

        # Validation loop
        with torch.no_grad():
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

                    training_logger.save_validation(
                        pred=pred,
                        labels=y,
                        loss=loss_fn(pred, y.long()).item(),
                    )
                    p.update(task, advance=1)

        training_logger.log_validation(epoch=epoch, model=model)
    return None
