"""DatasetStore.add(..., provenance=...): the optional provenance is saved in meta.json, and an add without it is unchanged."""
import json

import pytest

from prophet_agent.datasets import DatasetStore

PLAIN_FIELDS = {"id", "name", "bytes", "sha256", "created"}


def test_add_without_provenance_keeps_the_same_meta_fields(tmp_path):
    store = DatasetStore(tmp_path / "datasets")
    meta = store.add("sales.csv", b"week,units\n1,2\n")
    assert set(meta) == PLAIN_FIELDS
    assert "provenance" not in meta
    on_disk = json.loads((store.root / meta["id"] / "meta.json").read_text(encoding="utf-8"))
    assert set(on_disk) == PLAIN_FIELDS
    assert on_disk["name"] == "sales.csv" and on_disk["bytes"] == 15


def test_add_with_provenance_saves_it_inside_meta_json(tmp_path):
    store = DatasetStore(tmp_path / "datasets")
    provenance = {"source": "ecb_sdmx", "url": "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata",
                  "params": {"flow": "EXR", "key": "D.USD.EUR.SP00.A"}, "bytes": 15}
    meta = store.add("ecb.csv", b"week,units\n1,2\n", provenance=provenance)
    assert meta["provenance"] == provenance
    on_disk = json.loads((store.root / meta["id"] / "meta.json").read_text(encoding="utf-8"))
    assert on_disk["provenance"] == provenance
    assert set(on_disk) == PLAIN_FIELDS | {"provenance"}
    assert store.get(meta["id"])["provenance"] == provenance


def test_provenance_that_cannot_be_written_as_json_leaves_no_folder(tmp_path):
    store = DatasetStore(tmp_path / "datasets")
    with pytest.raises(TypeError):
        store.add("x.csv", b"a,b\n1,2\n", provenance={"when": object()})
    assert not (tmp_path / "datasets").exists() or list((tmp_path / "datasets").iterdir()) == []
