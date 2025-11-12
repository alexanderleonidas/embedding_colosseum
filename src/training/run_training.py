import logging

import pennylane as qml
import torch
from pennylane import numpy as np
from rich import print

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

    np.random.seed(0)
    torch.random.manual_seed(0)

    # weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    # bias_init = np.array(0.0, requires_grad=True)

    model = VariationalClassifier(
        num_qubits=cfg.model.num_qubits,
        num_layers=cfg.model.num_layers,
    )
    log.info(f"Weights: {model.weights}")
    log.info(f"Bias: {model.bias}")

    # opt = NesterovMomentumOptimizer(0.5)
    optimizer = torch.optim.AdamW([model.weights, model.bias], lr=0.1)
    batch_size = 5

    # Training Loop
    for it in range(cfg.training.steps):
        # Update the weights by one optimizer step, using only a limited batch of data
        batch_index = torch.randint(0, len(X), (batch_size,))
        X_batch = X[batch_index]
        Y_batch = Y[batch_index]

        # Get the loss
        def closure():
            optimizer.zero_grad()
            loss = model.cost(X_batch, Y_batch)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Compute accuracy
        predictions = [torch.sign(model.classify(x)) for x in X]

        current_cost = model.cost(X, Y)
        acc = accuracy(Y, predictions)

        log.info(
            f"Iter: {it + 1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}"
        )

    data = np.loadtxt("variational_classifier/data/parity_test.txt", dtype=int)
    X_test = torch.tensor(np.array(data[:, :-1]), dtype=torch.double).to(device=device)
    Y_test = torch.tensor(np.array(data[:, -1]), dtype=torch.double).to(device=device)
    Y_test = Y_test * 2 - 1  # shift label from {0, 1} to {-1, 1}

    predictions_test = [torch.sign(model.classify(x)) for x in X_test]

    for x, y, p in zip(X_test, Y_test, predictions_test):
        log.info(f"x = {x}, y = {y}, pred={p}")

    acc_test = accuracy(Y_test, predictions_test)
    log.info(f"Accuracy on unseen data: {acc_test}")

    # TODO add correct validation loss for hyperparameter tuning
    val_loss = current_cost
    return val_loss
