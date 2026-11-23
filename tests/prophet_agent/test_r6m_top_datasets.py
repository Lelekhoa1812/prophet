"""Round 6m: the uploaded-file store survives a damaged metadata file. Offline: every folder is a temporary one."""
from __future__ import annotations

import pytest

from prophet_agent.datasets import DatasetStore


def test_a_truncated_metadata_file_is_skipped_by_the_listing_not_fatal(tmp_path):
    store = DatasetStore(tmp_path / "datasets")
    good = store.add("sales.csv", b"day,units\n1,4\n")
    broken = tmp_path / "datasets" / "0123456789abcdef0123456789abcdef"
    broken.mkdir()
    (broken / "meta.json").write_text('{"id": "0123')            # a crash part-way through writing it
    listed = store.list()
    assert [m["id"] for m in listed] == [good["id"]]


def test_a_truncated_metadata_file_reads_as_an_unknown_dataset(tmp_path):
    store = DatasetStore(tmp_path / "datasets")
    did = "0123456789abcdef0123456789abcdef"
    (tmp_path / "datasets" / did).mkdir(parents=True)
    (tmp_path / "datasets" / did / "meta.json").write_text("{not json")
    with pytest.raises(KeyError):
        store.get(did)


def test_an_upload_leaves_only_its_file_and_its_metadata_in_its_folder(tmp_path):
    store = DatasetStore(tmp_path / "datasets")
    meta = store.add("sales.csv", b"day,units\n1,4\n")
    assert sorted(p.name for p in (tmp_path / "datasets" / meta["id"]).iterdir()) == ["meta.json", "sales.csv"]
