import logging

import hydra
from mda.data import DATASET_REGISTRY, MultiDomainDataset
from mda.data.data_collection import DataCollection
from mda.logreg import eval_logreg_model, train_logreg_model
from mda.model import MODEL_REGISTRY, Model
from mda.util import get_full_path, load_json, save_json, set_random_seed
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    logger.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")

    # load data collection
    data_collection_json_path = get_full_path(config.data_collection.path)
    logger.info(f"loading data collection from {data_collection_json_path}")
    data_collection_dict = load_json(data_collection_json_path)
    data_collection = DataCollection.parse_obj(data_collection_dict)
    logger.info(f"loaded data collection of {len(data_collection.samples)} sample")
    n_classes = len(data_collection.class_strs)
    logger.info(f"loaded data collection of {n_classes} classes")
    n_domains = len(data_collection.domain_strs)
    logger.info(f"loaded data collection of {n_domains} domains")

    # create dataset
    dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
        config.dataset.name,
        config.dataset.args,
        collection=data_collection,
        class_distribution_use_split="train",
    )
    logger.info(f"built dataset of {len(dataset.filtered_samples)} samples")

    # build model
    model: Model = MODEL_REGISTRY.from_config(
        config.model.name,
        config.model.args,
        n_classes=n_classes,
        n_domains=n_domains,
    )
    logger.info(f"built model {model}")

    # train
    train_logreg_model(
        model,
        dataset,
        num_epoch=20,  # fixme
    )
    print(eval_logreg_model(model, dataset))


if __name__ == "__main__":
    main()
