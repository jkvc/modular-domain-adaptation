import logging

import hydra
from mda.data import DATASET_REGISTRY, MultiDomainDataset
from mda.data.data_collection import DataCollection
from mda.logreg import train_logreg_model
from mda.model import MODEL_REGISTRY, Model
from mda.util import (AUTO_DEVICE, is_experiment_done, load_json,
                      mark_experiment_done, mkdirs, save_json)
from omegaconf import OmegaConf
from repo_root import get_full_path

from experiments.acc import compute_accs
from experiments.trainer import TRAINER_REGISTRY

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )
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

    for use_domain_str in train_collection.domain_strs:
        logger.info(f"running experiment using only [{use_domain_str}]")
        holdout_domain_strs = [
            d for d in train_collection.domain_strs if d != use_domain_str
        ]
        logger.info(f"use domain {holdout_domain_strs}")

        # output_dir
        output_dir = get_full_path(
            f"{config.working_dir}/single_domain/{config.data_collection.name}/{config.model.arch}/{use_domain_str}"
        )
        if is_experiment_done(output_dir):
            continue
        logger.info(f"saving all outputs to {output_dir}")
        mkdirs(output_dir, overwrite=True)

        # create dataset
        train_dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
            config.dataset.name,
            config.dataset.args,
            collection=train_collection,
            use_domain_strs=[use_domain_str],
        )
        logger.info(
            f"built train dataset of {len(train_dataset.filtered_samples)} samples"
        )
        test_dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
            config.dataset.name,
            config.dataset.args,
            collection=train_collection,
            use_domain_strs=holdout_domain_strs,
            **train_dataset.computed_asset(),
        )
        logger.info(
            f"built test dataset of {len(test_dataset.filtered_samples)} samples"
        )

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
