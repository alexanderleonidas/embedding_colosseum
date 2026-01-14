import logging
import os
from copy import copy

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


@torch.no_grad()  # Ensure no gradients are computed during accuracy calculation
def accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    num_correct = (labels == predictions).sum().item()
    return num_correct / len(labels)


class TrainingLogger:
    """The TrainingLogger is a utility class not only to print and save metrics but also
    to save the model weights and biases. This combines everything so that the code with
    the training loop is more clean.
    """

    def __init__(self, cfg, train_loader, val_loader):
        self.cfg = cfg

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.artifact_y_true, self.artifact_y_pred = [], []
        self.val_acc, self.val_loss = 0.0, 0.0

        self.root = HydraConfig.get().runtime.output_dir

        # Initialize csv for training results
        self.csv_fp = os.path.join(self.root, "results.csv")
        with open(self.csv_fp, "w") as f:
            f.write("epoch,batch,train_loss,train_acc,val_loss,val_acc\n")

    # Procedure to save a trained model
    def save_model_weights_and_bias(self, weights, bias, epoch):
        filename = os.path.join(self.root, f"saved_models/model_epoch-{epoch}.pt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({"weights": copy(weights), "bias": copy(bias)}, filename)
        log.info(f"Model dict saved to {filename}")
        pass

    # Procedure to save training data
    def log_training(self, pred, labels, loss, epoch, batch):
        acc = accuracy(pred, labels)
        current = batch * self.cfg.training.batch_size
        with open(self.csv_fp, "a") as f:
            f.write(f"{epoch},{batch},{loss},{acc},,\n")

            log.info(
                f"[{current:>5d}/{len(self.train_loader.dataset):>5d}] | "
                + f"Cost: {loss:0.7f} | Accuracy: {acc:0.7f}"
            )
        pass

    def save_validation(self, pred, labels, loss):
        """Save metrics and raw results into RAM during the validation loop. Logging
        happens in another function.
        """
        # Save logits for artifact
        self.val_acc += accuracy(logits=pred, labels=labels)
        self.val_loss += loss

        self.artifact_y_true.extend(labels.cpu().numpy())
        self.artifact_y_pred.extend(pred.cpu().numpy())
        pass

    def log_validation(self, epoch, model):
        """Save the stored results from save_validation into persistent storage and
        print out a summary.
        """
        self.save_model_weights_and_bias(model.weights, model.bias, epoch)

        self.val_loss /= len(self.val_loader)
        self.val_acc /= len(self.val_loader)
        log.info(
            f"Validation for epoch {epoch:>3} | Cost: {self.val_loss:0.7f} | Accuracy: {self.val_acc:0.7f}"
        )
        with open(self.csv_fp, "a") as f:
            f.write(
                f"{epoch},{len(self.train_loader)},,,{self.val_loss},{self.val_acc}\n"
            )

        # Save raw results for later analysis
        np.savez_compressed(
            os.path.join(self.root, "raw_results.npz"),
            y_true=self.artifact_y_true,
            y_pred=self.artifact_y_pred,
        )

        # Reset values for next validation loop
        self.artifact_y_true, self.artifact_y_pred = [], []
        self.val_acc, self.val_loss = 0.0, 0.0
