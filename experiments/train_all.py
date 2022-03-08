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

    # load train collection
    train_collection_json_path = get_full_path(config.data_collection.train_path)
    logger.info(f"loading train collection from {train_collection_json_path}")
    train_collection_dict = load_json(train_collection_json_path)
    train_collection = DataCollection.parse_obj(train_collection_dict)
    logger.info(f"loaded train collection of {len(train_collection.samples)} sample")
    n_classes = len(train_collection.class_strs)
    logger.info(f"loaded train collection of {n_classes} classes")
    n_domains = len(train_collection.domain_strs)
    logger.info(f"loaded train collection of {n_domains} domains")

    # create dataset
    train_dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
        config.dataset.name,
        config.dataset.args,
        collection=train_collection,
    )
    logger.info(f"built train dataset of {len(train_dataset.filtered_samples)} samples")

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
        train_dataset,
        num_epoch=20,  # fixme
    )

    # test collection
    test_collection_json_path = get_full_path(config.data_collection.test_path)
    logger.info(f"loading test collection from {test_collection_json_path}")
    test_collection_dict = load_json(test_collection_json_path)
    test_collection = DataCollection.parse_obj(test_collection_dict)
    logger.info(f"loaded test collection of {len(test_collection.samples)} sample")
    test_dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
        config.dataset.name,
        config.dataset.args,
        collection=test_collection,
        vocab_override=train_dataset.vocab,
    )
    logger.info(f"built test dataset of {len(test_dataset.filtered_samples)} samples")

    print(eval_logreg_model(model, test_dataset))


if __name__ == "__main__":
    main()
