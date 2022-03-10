import logging

import hydra
from mda.data import DATASET_REGISTRY, MultiDomainDataset
from mda.data.data_collection import DataCollection
from mda.logreg import train_logreg_model
from mda.model import MODEL_REGISTRY, Model
from mda.util import (
    AUTO_DEVICE,
    is_experiment_done,
    load_json,
    mark_experiment_done,
    mkdirs,
    save_json,
)
from omegaconf import OmegaConf
from repo_root import get_full_path

from experiments.acc import compute_accs
from experiments.trainer import TRAINER_REGISTRY

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    logger.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")

    # output_dir
    output_dir = get_full_path(
        f"{config.working_dir}/all_domain/{config.data_collection.name}/{config.model.arch}"
    )
    if is_experiment_done(output_dir):
        return
    logger.info(f"saving all outputs to {output_dir}")
    mkdirs(output_dir, overwrite=True)

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
    # test collection
    test_collection_json_path = get_full_path(config.data_collection.test_path)
    logger.info(f"loading test collection from {test_collection_json_path}")
    test_collection_dict = load_json(test_collection_json_path)
    test_collection = DataCollection.parse_obj(test_collection_dict)
    logger.info(f"loaded test collection of {len(test_collection.samples)} sample")

    # create dataset
    train_dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
        config.dataset.name,
        config.dataset.args,
        collection=train_collection,
    )
    logger.info(f"built train dataset of {len(train_dataset.filtered_samples)} samples")
    test_dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
        config.dataset.name,
        config.dataset.args,
        collection=test_collection,
        **train_dataset.computed_asset(),
    )
    logger.info(f"built test dataset of {len(test_dataset.filtered_samples)} samples")

    # build model
    model: Model = MODEL_REGISTRY.from_config(
        config.model.name,
        config.model.args,
        n_classes=n_classes,
        n_domains=n_domains,
    ).to(AUTO_DEVICE)
    logger.info(f"built model {model}")

    # train
    trainer = TRAINER_REGISTRY.from_config(
        config.trainer,
        {},
        model=model,
        dataset=train_dataset,
    )
    trainer.run()

    # log model
    model.to_logdir(output_dir)

    # save acc
    acc = compute_accs(model, train_dataset, test_dataset)
    save_json(acc.dict(), f"{output_dir}/acc.json")

    mark_experiment_done(output_dir)


if __name__ == "__main__":
    main()
