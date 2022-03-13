# data_raw

This directory is to contain the raw data before any data preprocessing, as used in the experiments.

## Data sources

In our experiment, all data is retrieved in March 2021.

### amazon

We use the some categories form the popular Amazon Review dataset by Ni et. al. Please download the relevant categories' 5-core from the [official published source](https://nijianmo.github.io/amazon/).

### arxiv

We use the Arxiv dataset as published on [Kaggle](https://www.kaggle.com/Cornell-University/arxiv) by Cornell university. Please download the archive and extract the json.

### mfc

We use the Media Frame Corpus by Card et. al. 
TODO Dallas please fill this in.

### senti

We use a collection of sentiment review datasets from the following sources.

- Twitter US Airline Sentiment [link](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- Amazon books review 5-core [link](https://nijianmo.github.io/amazon/)
- Large Movie Review Dataset (IMDb) [link](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Sentiment 140 [link](http://help.sentiment140.com/for-students)
- Stanford Sentiment Treebank [link](https://nlp.stanford.edu/sentiment/code.html)

## Extraction

Please extract the relevant files to the following directory structure in `repo_root/data_raw`:

```
├── amazon
│   ├── Clothing_Shoes_and_Jewelry_5.json.gz
│   ├── Electronics_5.json.gz
│   ├── Home_and_Kitchen_5.json.gz
│   ├── Kindle_Store_5.json.gz
│   └── Movies_and_TV_5.json.gz
├── arxiv
│   └── arxiv-metadata-oai-snapshot.json
├── mfc
│   ├── climate_labeled.json
│   ├── climate_test_sets.json
│   ├── deathpenalty_labeled.json
│   ├── deathpenalty_test_sets.json
│   ├── guncontrol_labeled.json
│   ├── guncontrol_test_sets.json
│   ├── immigration_labeled.json
│   ├── immigration_test_sets.json
│   ├── police_labeled.json
│   ├── police_test_sets.json
│   ├── samesex_labeled.json
│   ├── samesex_test_sets.json
│   ├── tobacco_labeled.json
│   └── tobacco_test_sets.json
├── readme.md
└── senti
    ├── airline
    │   └── Tweets.csv
    ├── amazon
    │   └── Books_5.json.gz
    ├── imdb
    │   └── IMDB\ Dataset.csv
    ├── senti140
    │   └── training.1600000.processed.noemoticon.csv
    └── sst
        ├── datasetSentences.txt
        └── sentiment_labels.txt
```
