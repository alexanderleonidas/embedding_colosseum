import logging
import time

import pennylane as qml
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pennylane import numpy as np
from rich import print

from src.dataset.DataManager import DataManager
from src.embeddings.FRQI_PennyLane import FRQI
from src.embeddings.NEQR_PennyLane import NEQR
from src.model.VariationalClassifier import VariationalClassifier
from src.utils.save_training_progress import TrainingLogger

log = logging.getLogger(__name__)
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def accuracy(labels, predictions):
    acc = 0.0
    for label, pred in zip(labels, predictions):
        acc += torch.sum(torch.abs(label - pred) < 1e-5)
    acc = acc / len(labels)
    return acc.item()


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
    else:
        raise ValueError("Unknown embedding method")

    dm = DataManager(
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
        dataset="mnist_binary",
        pixel_size=cfg.training.image_width,  # pixel size set above
    )
    train_loader, validation_loader, test_loader = dm.get_loaders(
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
    )
    # TODO Adjust model for number of output classes:
    # UserWarning: Using a target size (torch.Size([10])) that is different to the input size (torch.Size([1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
    #   return F.mse_loss(input, target, reduction=self.reduction)
    model = VariationalClassifier(
        num_qubits=embedding.num_qubits,
        num_layers=cfg.model.num_layers,
        num_classes=2,
        num_pixels=cfg.training.image_width * cfg.training.image_width,
        state_preparation=embedding.state_preparation,  # embedding set above in run_classifier
        # state_preparation=NEQR(num_pixels=2).state_preparation,  # TODO parameterize
    )
    log.info(f"Weights Shape: {model.weights.shape}")
    log.info(f"Bias Shape: {model.bias.shape}")

    optimizer = torch.optim.AdamW([model.weights, model.bias], lr=0.1)

    # Training Loop
    total_samples = len(train_loader.dataset)
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            # Computes the loss
            def closure():
                optimizer.zero_grad()
                # Ensure tensors are the expected dtype for autograd
                loss = model.cost(X.to(device).double(), y.to(device).double())
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            current_cost = loss.item()
            epoch_loss += current_cost

            # Compute accuracy
            predictions = torch.stack([torch.sign(model.classify(x)) for x in X])
            acc = accuracy(y, predictions)
            epoch_acc += acc
            current = batch * cfg.training.batch_size

            log.info(
                f"[{current:>5d}/{total_samples:>5d}] | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}"
            )

            # Save training results per batch
            with open(csv_fp, "a") as f:
                f.write(f"{epoch},{batch},{current_cost},{acc},,\n")

        # Accumulate epoch metrics
        epoch_loss /= total_samples
        epoch_loss /= total_samples
        # Validation loop
        val_acc, val_loss = 0.0, 0.0
        for X, y in validation_loader:
            X = X.to(device).double()
            y = y.to(device).double()

            val_loss += model.cost(X, y).item()

            preds = torch.stack([torch.sign(model.classify(x)) for x in X])
            val_acc += accuracy(y, preds)

        val_loss /= len(validation_loader)
        val_acc /= len(validation_loader)
        log.info(
            f"Validation for epoch {epoch:>3} | Cost: {val_loss:0.7f} | Accuracy: {val_acc:0.7f}"
        )
        with open(csv_fp, "a") as f:
            f.write(f"{epoch},,,,{val_loss},{val_acc}\n")

        metrics_to_save = {
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "epoch_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        # training_logger.save_training_progression(metrics_to_save)
        training_logger.save_model_weights_and_bias(model.weights, model.bias, epoch)

    return val_loss
