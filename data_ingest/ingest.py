import logging

import hydra
from omegaconf import OmegaConf

from data_ingest.ingestor import INGESTOR_REGISTRY

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="ingest")
def main(config: OmegaConf):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    logger.info(OmegaConf.to_yaml(config, resolve=True))

    ingestor = INGESTOR_REGISTRY.from_config(config.ingestor.name, {})
    logger.info(f"ingestor {ingestor}")

    ingestor.run()


if __name__ == "__main__":
    main()
