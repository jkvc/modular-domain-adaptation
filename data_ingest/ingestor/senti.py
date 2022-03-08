import base64
import gzip
import hashlib
import json
import logging
import random
from typing import List

import pandas as pd
from mda.data.data_collection import DataCollection, DataSample, create_random_split
from mda.util import get_full_path, read_txt_as_str_list
from tqdm import tqdm

from . import INGESTOR_REGISTRY, Ingestor

logger = logging.getLogger(__name__)
RAW_DIR = "data_raw/senti"
POLARITY_NAMES = ["neg", "pos"]
SENTIMENT_SOURCES = [
    "airline",
    "amazon",
    "imdb",
    "senti140",
    "sst",
]
N_SAMPLE_PER_DOMAIN = 10000


def ingest_airline():
    logger.info("ingesting airline tweets")
    samples = {}
    df = pd.read_csv(get_full_path(f"{RAW_DIR}/airline/Tweets.csv"))
    idxs = random.sample(range(len(df)), N_SAMPLE_PER_DOMAIN)
    for idx in idxs:
        row = df.iloc[idx]
        tweet_id = row[0]
        polarity = row[1]
        text = row[10]
        id = f"airline.{tweet_id}"
        if polarity == "positive":
            class_idx = 1
        elif polarity == "negative":
            class_idx = 0
        else:
            continue
        sample = DataSample(
            id=id,
            text=text,
            domain_str="airline",
            class_str=POLARITY_NAMES[class_idx],
            class_idx=class_idx,
        )
        samples[id] = sample
    return list(samples.values())


def ingest_amazon():
    logger.info("ingesting amazon books")
    TOTAL_LINES_READ = 1600000
    idxs = set(random.sample(range(TOTAL_LINES_READ), N_SAMPLE_PER_DOMAIN))
    raw_samples = []
    g = gzip.open(get_full_path(f"{RAW_DIR}/amazon/Books_5.json.gz"), "r")
    for i, l in enumerate(g):
        if i > TOTAL_LINES_READ:
            break
        if i in idxs:
            raw_samples.append(json.loads(l))
    samples = {}
    for raw_sample in raw_samples:
        if "reviewText" not in raw_sample or "overall" not in raw_sample:
            continue
        text = raw_sample["reviewText"]
        rating = int(raw_sample["overall"])
        if rating <= 2:
            class_idx = 0
        elif rating >= 4:
            class_idx = 1
        else:
            continue
        new_id = f"amazon.{raw_sample['asin']}-{raw_sample['reviewerID']}"
        sample = DataSample(
            id=new_id,
            text=text,
            domain_str="amazon",
            class_str=POLARITY_NAMES[class_idx],
            class_idx=class_idx,
        )
        samples[new_id] = sample
    return list(samples.values())


def ingest_imdb():
    logger.info("ingesting imdb reviews")
    df = pd.read_csv(get_full_path(f"{RAW_DIR}/imdb/IMDB Dataset.csv"))
    idxs = random.sample(range(len(df)), N_SAMPLE_PER_DOMAIN)
    samples = {}
    for idx in idxs:
        row = df.iloc[idx]
        text = row[0]
        polarity = row[1]
        if polarity == "positive":
            class_idx = 1
        elif polarity == "negative":
            class_idx = 0
        else:
            continue

        hasher = hashlib.sha1(text.encode())
        review_id = base64.urlsafe_b64encode(hasher.digest()[:6]).decode()
        new_id = f"imdb.{review_id}"
        sample = DataSample(
            id=new_id,
            text=text,
            domain_str="imdb",
            class_str=POLARITY_NAMES[class_idx],
            class_idx=class_idx,
        )
        samples[new_id] = sample
    return list(samples.values())


def ingest_senti140():
    logger.info("ingesting sentiment 140 dataset")
    df = pd.read_csv(
        get_full_path(f"{RAW_DIR}/senti140/training.1600000.processed.noemoticon.csv")
    )
    idxs = random.sample(range(len(df)), N_SAMPLE_PER_DOMAIN)
    samples = {}
    for idx in idxs:
        row = df.iloc[idx]
        tweet_id = row[1]
        text = row[5]
        polarity = row[0]
        if polarity == 0:
            class_idx = 0
        elif polarity == 4:
            class_idx = 1
        else:
            continue
        new_id = f"senti140.{tweet_id}"
        sample = DataSample(
            id=new_id,
            text=text,
            domain_str="senti140",
            class_str=POLARITY_NAMES[class_idx],
            class_idx=class_idx,
        )
        samples[new_id] = sample
    return list(samples.values())


def ingest_sst():
    logger.info("ingesting stanford sentiment treebank")

    # process labels
    lines = read_txt_as_str_list(get_full_path(f"{RAW_DIR}/sst/sentiment_labels.txt"))
    id2sentiscore = {}
    for l in lines[1:]:
        phrase_id, sentiscore = l.split("|")
        id2sentiscore[int(phrase_id)] = float(sentiscore)

    # process samples
    df = pd.read_csv(get_full_path(f"{RAW_DIR}/sst/datasetSentences.txt"), sep="\t")
    samples = {}
    for _, row in df.iterrows():
        phrase_id = row[0]
        text = row[1]
        sentiscore = id2sentiscore[phrase_id]
        if sentiscore < 0.3:
            class_idx = 0
        elif sentiscore > 0.7:
            class_idx = 1
        else:
            continue
        new_id = f"sst.{phrase_id}"
        sample = DataSample(
            id=new_id,
            text=text,
            domain_str="sst",
            class_str=POLARITY_NAMES[class_idx],
            class_idx=class_idx,
        )
        samples[new_id] = sample
    return list(samples.values())


@INGESTOR_REGISTRY.register("senti")
class SentimentCollectionIngestor(Ingestor):
    def run(self) -> DataCollection:
        collection = DataCollection()

        samples = []
        samples.extend(ingest_airline())
        samples.extend(ingest_amazon())
        samples.extend(ingest_imdb())
        samples.extend(ingest_senti140())
        samples.extend(ingest_sst())

        for sample in samples:
            collection.add_sample(sample)

        all_train_ids, all_test_ids = create_random_split(
            list(collection.samples.keys())
        )

        train_collection = DataCollection()
        for id in all_train_ids:
            train_collection.add_sample(collection.samples[id])
        train_collection.class_strs = POLARITY_NAMES
        train_collection.domain_strs = SENTIMENT_SOURCES
        train_collection.populate_class_distribution()

        test_collection = DataCollection()
        for id in all_test_ids:
            test_collection.add_sample(collection.samples[id])
        test_collection.class_strs = POLARITY_NAMES
        test_collection.domain_strs = SENTIMENT_SOURCES
        test_collection.populate_class_distribution()

        return train_collection, test_collection
