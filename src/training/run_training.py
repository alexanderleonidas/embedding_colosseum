import logging

import pennylane as qml
import torch
from pennylane import numpy as np
from rich import print

from src.dataset.DataManager import DataManager
from src.embeddings.FRQI_PennyLane import FRQI
from src.embeddings.NEQR_PennyLane import NEQR
from src.model.VariationalClassifier import VariationalClassifier

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
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    log.info(f"Using {device} to train.")

    data = np.loadtxt("variational_classifier/data/parity_train.txt", dtype=int)
    X = np.array(data[:, :-1])
    Y = np.array(data[:, -1])
    Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}

    # If at least one input feature is on cuda, the computation will be done on cuda
    X = torch.tensor(X, dtype=torch.double).to(device=device)
    Y = torch.tensor(Y, dtype=torch.double).to(device=device)
    log.info(X.shape)
    log.info(Y.shape)

    for x, y in zip(X, Y):
        log.info(f"x = {x}, y = {y}")

    dm = DataManager(
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
        dataset="mnist",
        pixel_size=32,  # Set the pixels to 32x32
    )
    train_loader, validation_loader, test_loader = dm.get_loaders()

    model = VariationalClassifier(
        num_qubits=cfg.model.num_qubits,
        num_layers=cfg.model.num_layers,
        # state_preparation=FRQI(num_pixels=2).state_preparation,  # TODO parameterize
        state_preparation=NEQR(num_pixels=2).state_preparation,  # TODO parameterize
    )
    log.info(f"Weights: {model.weights}")
    log.info(f"Bias: {model.bias}")

    optimizer = torch.optim.AdamW([model.weights, model.bias], lr=0.1)

    # Training Loop
    train_size = len(train_loader)
    for epoch in range(cfg.training.epochs):
        for batch, (X, y) in enumerate(train_loader):
            # Get the loss
            def closure():
                optimizer.zero_grad()
                loss = model.cost(X.to(device), y.to(device))
                loss.backward()
                return loss

        optimizer.step(closure)

        # Compute accuracy
        predictions = [torch.sign(model.classify(x)) for x in X]

        current_cost = model.cost(X, Y)
        acc = accuracy(Y, predictions)

        current = batch * len(X)
        log.info(
            f"[{current:>5d}/{train_size:>5d}] | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}"
        )

        # Validation loop
        val_acc, val_loss = 0, 0
        for batch, (X, y) in enumerate(validation_loader):
            val_loss += model.cost(X)
            val_acc += accuracy(
                labels=y, predictions=torch.sign(model.classify(X))
            ) / len(X)  # Normalize the accuracy

        val_loss = val_loss / len(validation_loader)
        log.info(
            f"Validation for epoch {epoch:>3} | Cost: {val_loss:0.7f} | Accuracy: {val_acc:0.7f}"
        )
    return val_loss
