# data_ingest

Please first follow the raw data retrieval guide in [../data_raw](../data_raw), and install the necessary prerequisite in [../experiments](../experiments).

Then run the ingestor to ingest all data to preprocess them to our defined data formats.

```
python ingest.py -m 'ingestor=glob(*)'
```

To ingest new dataset, either define an ingestor following the example, or save the data in the format as defined in [DataCollection.py](../mda/data/data_collection.py) to be loaded by the main API.
