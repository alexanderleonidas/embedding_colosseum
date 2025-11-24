import os
from copy import copy

import torch
from omegaconf import OmegaConf


class TrainingLogger:
    def __init__(self, run_id, cfg):
        self.run_id = run_id
        self.cfg = cfg
        self.experiment_dir = self._get_exp_dir()
        try:
            self.save_cfg()
        except Exception:
            pass

    def save_cfg(self):
        filename = self._get_filename('.yaml', 'config')
        OmegaConf.save(self.cfg, filename)
        print(f'Config saved to {filename}')

    # Procedure to save a trained model
    def save_model_weights_and_bias(self, weights, bias, epoch):
        filename = self._get_filename('.pt', f'saved_models/model_epoch-{epoch}')
        copy_dict = {'weights': copy(weights), 'bias': copy(bias)}
        torch.save(copy_dict, filename)
        print(f'Model dict saved to {filename}')


    # Procedure to save training data
    def save_training_progression(self, metrics):
        filename = self._get_filename('.csv', 'training_progression')
        file_exists = os.path.exists(filename)
        with open(filename, 'a') as f:
            if not file_exists:
                f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')
            f.write(f"{metrics['epoch']},{metrics['epoch_loss']},{metrics['epoch_acc']},{metrics['val_loss']},{metrics['val_acc']}\n")


    # Procedure to save test data
    def save_test_progress(self, test_loss, exp_num=None):
        filename = self._get_filename('.csv', 'test_results')
        file_exists = os.path.exists(filename)
        with open(filename, 'a') as f:
            if exp_num is None:
                if not file_exists:
                    f.write('test_loss\n')
                f.write(f"{test_loss}\n")
            else:
                if not file_exists:
                    f.write('run,test_loss\n')
                f.write(f"{exp_num},{test_loss}\n")

    def _get_filename(self, extension, addition=''):
        if extension[0] != '.' or extension == '' or extension is None:
            raise ValueError('Please provide a valid file extension with a leading dot.')
        return os.path.join(self.experiment_dir, f"{addition}{extension}")

    # Function to generate a unique filename for a saved model
    def _get_exp_dir(self):
        root = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]
        file_dir = os.path.join(root, 'results')
        os.makedirs(file_dir, exist_ok=True)
        exp_dir = os.path.join(file_dir, f"run_{self.run_id}")
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'saved_models'), exist_ok=True)
        return exp_dir

# if __name__ == 'main':
print('exp_dir: ', TrainingLogger(122, 0).save_model_weights_and_bias(torch.tensor([1,2,3]), torch.tensor([4,5,6]), 1))
