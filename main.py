import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.model.VariationalClassifier import VariationalClassifier
from src.training.run_training import run_classifier

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    loss = run_classifier(cfg=cfg)
    return loss


if __name__ == "__main__":
    main()
