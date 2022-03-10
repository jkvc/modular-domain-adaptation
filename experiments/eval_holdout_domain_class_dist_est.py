import logging
import random

import hydra
from mda.data import DATASET_REGISTRY, MultiDomainDataset
from mda.data.bow_dataset import get_vocab_from_lexicon_csv
from mda.data.data_collection import DataCollection, compute_class_distribution
from mda.logreg import train_logreg_model
from mda.model import MODEL_REGISTRY, Model
from mda.util import (
    AUTO_DEVICE,
    get_full_path,
    is_experiment_done,
    load_json,
    mark_experiment_done,
    mkdirs,
    save_json,
)
from omegaconf import OmegaConf

from experiments.acc import compute_accs
from experiments.trainer import TRAINER_REGISTRY

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="class_dist_estimation")
def main(config: OmegaConf):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    logger.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")

    # load train collection, use holdout source as OOD samples
    train_collection_json_path = get_full_path(config.data_collection.train_path)
    logger.info(f"loading train collection from {train_collection_json_path}")
    train_collection_dict = load_json(train_collection_json_path)
    train_collection = DataCollection.parse_obj(train_collection_dict)
    logger.info(f"loaded train collection of {len(train_collection.samples)} sample")
    n_classes = len(train_collection.class_strs)
    logger.info(f"loaded train collection of {n_classes} classes")
    n_domains = len(train_collection.domain_strs)
    logger.info(f"loaded train collection of {n_domains} domains")

    for holdout_domain in train_collection.domain_strs:
        logger.info(f"evaluating using [{holdout_domain}]")

        # load model from checkpoint
        model_checkpoint_dir = get_full_path(
            f"{config.working_dir}/holdout_domain/{config.data_collection.name}/{config.model.arch}/{holdout_domain}"
        )
        logger.info(f"loading model from checkpoint {model_checkpoint_dir}")

        for trial_idx in range(config.n_trial):
            # output_dir
            output_dir = get_full_path(
                f"{config.working_dir}/holdout_domain_class_dist_est/{config.data_collection.name}/{config.model.arch}/{config.n_labeled_samples}/{holdout_domain}/{trial_idx}"
            )
            if is_experiment_done(output_dir):
                continue
            logger.info(f"saving all outputs to {output_dir}")
            mkdirs(output_dir, overwrite=True)

            # compute class distribution override
            candidate_samples = [
                s
                for s in train_collection.samples.values()
                if s.domain_str == holdout_domain
            ]
            chosen_samples = random.sample(
                candidate_samples, k=config.n_labeled_samples
            )
            class_distribution_override = compute_class_distribution(
                chosen_samples, len(train_collection.class_strs)
            )

            # eval dataset
            eval_dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
                config.dataset.name,
                config.dataset.args,
                collection=train_collection,
                use_domain_strs=[holdout_domain],
                class_distribution_override=class_distribution_override,
                vocab_override=get_vocab_from_lexicon_csv(model_checkpoint_dir),
            )

            # load model
            model: Model = MODEL_REGISTRY.from_config(
                config.model.name,
                config.model.args,
                n_classes=n_classes,
                n_domains=n_domains,
            ).to(AUTO_DEVICE)
            logger.info("loading model from logdir")
            model.from_logdir(model_checkpoint_dir)

            # eval, save acc
            acc = compute_accs(model, None, eval_dataset)
            save_json(acc.dict(), f"{output_dir}/acc.json")

            mark_experiment_done(output_dir)


if __name__ == "__main__":
    main()
