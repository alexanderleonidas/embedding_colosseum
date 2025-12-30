import os
from copy import copy

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


class TrainingLogger:
    def __init__(self, run_id, cfg):
        self.run_id = run_id
        self.cfg = cfg
        self.experiment_dir = self._get_exp_dir()

    # Procedure to save a trained model
    def save_model_weights_and_bias(self, weights, bias, epoch):
        filename = self._get_filename(".pt", f"saved_models/model_epoch-{epoch}")
        copy_dict = {"weights": copy(weights), "bias": copy(bias)}
        torch.save(copy_dict, filename)
        print(f"Model dict saved to {filename}")

    # Procedure to save training data
    def save_training_progression(self, metrics):
        filename = self._get_filename(".csv", "training_progression")
        file_exists = os.path.exists(filename)
        with open(filename, "a") as f:
            if not file_exists:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
            f.write(
                f"{metrics['epoch']},{metrics['epoch_loss']},{metrics['epoch_acc']},{metrics['val_loss']},{metrics['val_acc']}\n"
            )

    # Procedure to save test data
    def save_test_progress(self, test_loss, exp_num=None):
        filename = self._get_filename(".csv", "test_results")
        file_exists = os.path.exists(filename)
        with open(filename, "a") as f:
            if exp_num is None:
                if not file_exists:
                    f.write("test_loss\n")
                f.write(f"{test_loss}\n")
            else:
                if not file_exists:
                    f.write("run,test_loss\n")
                f.write(f"{exp_num},{test_loss}\n")

    def _get_filename(self, extension, addition=""):
        if extension[0] != "." or extension == "" or extension is None:
            raise ValueError(
                "Please provide a valid file extension with a leading dot."
            )
        return os.path.join(self.experiment_dir, f"{addition}{extension}")

    # Function to generate a unique filename for a saved model
    def _get_exp_dir(self):
        return HydraConfig.get().runtime.output_dir


# Example usage
# tl = TrainingLogger(122, 0)
# tl.save_model_weights_and_bias(torch.tensor([1,2,3]), torch.tensor([4,5,6]), 1)
# tl.save_training_progression({'epoch': 1, 'epoch_loss': 1, 'epoch_acc': 1, 'val_loss': 1, 'val_acc': 1})
# tl.save_test_progress(1)
