"""Open-data downloads for the forecasting agent: a source registry, a guarded fetch, and storage with provenance.

Motivation vs Logic
Motivation: the agent should be able to fetch public series (weather, ECB rates) without a person handing it a file, and
every stored series must say exactly where its bytes came from.
Logic: sources.py builds each URL from checked parameters, fetch.py downloads one URL under the network rules shared with
the research tools, and store.py puts the bytes into the dataset store with a provenance record. Nothing here reads a key
or a token; licences are named by key and documented in docs/prophet/DATASET_LICENCES.md.
"""
