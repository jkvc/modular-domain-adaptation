import logging
import random

import hydra
import torch
from mda.data import DATASET_REGISTRY, MultiDomainDataset
from mda.data.bow_dataset import get_vocab_from_lexicon_csv
from mda.data.data_collection import DataCollection, compute_class_distribution
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
from tqdm import tqdm

from experiments.acc import ModelAccuracy, compute_accs

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="class_dist_est")
def main(config: OmegaConf):
    config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )
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

        # load model
        model: Model = MODEL_REGISTRY.from_config(
            config.model.name,
            config.model.args,
            n_classes=n_classes,
            n_domains=n_domains,
        ).to(AUTO_DEVICE)
        logger.info("loading model from logdir")
        model.from_logdir(model_checkpoint_dir)

        # turn off dsbias, predict once, then add dsbias for each trial later
        model.use_domain_specific_bias = False
        model.eval()

        # eval dataset
        eval_dataset: MultiDomainDataset = DATASET_REGISTRY.from_config(
            config.dataset.name,
            config.dataset.args,
            collection=train_collection,
            use_domain_strs=[holdout_domain],
            vocab_override=get_vocab_from_lexicon_csv(model_checkpoint_dir),
        )

        loader = eval_dataset.get_loader()
        pred_logits = []
        pred_class_idx = []
        for batch in tqdm(loader):
            with torch.no_grad():
                batch = {
                    k: (v.to(AUTO_DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                pred_batch = model(batch)
                pred_logits.append(pred_batch["logits"])
                pred_class_idx.append(pred_batch["class_idx"])
        pred_logits = torch.cat(pred_logits, dim=0)
        pred_class_idx = torch.cat(pred_class_idx, dim=0)

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
            class_distribution_estimated = compute_class_distribution(
                chosen_samples, len(train_collection.class_strs)
            )[holdout_domain]
            class_distribution_estimated = torch.FloatTensor(
                class_distribution_estimated
            ).to(AUTO_DEVICE)
            trial_logits = pred_logits + torch.log(class_distribution_estimated)
            trial_pred = torch.argmax(trial_logits, dim=-1)
            is_correct = trial_pred == pred_class_idx
            num_correct = is_correct.sum()

            # eval, save acc
            acc = (num_correct / len(pred_class_idx)).item()
            acc = ModelAccuracy(train_acc=0, test_acc=acc)
            save_json(acc.dict(), f"{output_dir}/acc.json")

            mark_experiment_done(output_dir)


if __name__ == "__main__":
    main()
