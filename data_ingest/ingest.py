import logging

import hydra
from mda.data.data_collection import DataCollection
from mda.util import get_full_path, save_json, set_random_seed
from omegaconf import OmegaConf

from data_ingest.ingestor import INGESTOR_REGISTRY

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="ingest")
def main(config: OmegaConf):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    logger.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")

    set_random_seed(config.seed)

    ingestor = INGESTOR_REGISTRY.from_config(config.ingestor.name, {})
    logger.info(f"ingestor {ingestor}")

    collection: DataCollection = ingestor.run()
    dst_path = get_full_path(f"data/{config.ingestor.name}.json")
    save_json(collection.dict(), dst_path)

    logger.info(f"total num sample {len(collection.samples)}")
    logger.info(f"total num train sample {len(collection.split.train_ids)}")
    logger.info(f"total num  test sample {len(collection.split.test_ids)}")

    for domain_str in collection.domain_strs:
        nsamples = sum(
            [
                1
                for sample in collection.samples.values()
                if sample.domain_str == domain_str
            ]
        )
        logger.info(f"num sample in domain [{domain_str}] {nsamples}")

    logger.info(f"written to path: {dst_path}")


if __name__ == "__main__":
    main()
