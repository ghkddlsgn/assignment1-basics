import os

import pytest
import tiktoken
import torch
from datasets import Dataset
from datasets.table import MemoryMappedTable
from torch.utils.data import DataLoader

from cs336_basics.training_data import load_training_dataset


@pytest.fixture
def tokenizer():
    # Real BPE, with local ranks so tests never download tokenizer assets.
    return tiktoken.Encoding(
        name="offline-test",
        pat_str=r"\s+|[^\s]+",
        mergeable_ranks={**{bytes([i]): i for i in range(256)}, b"ab": 256},
        special_tokens={"<|endoftext|>": 257},
    )


def test_arrow_packing_matches_whole_corpus_and_shuffles(tmp_path, tokenizer):
    source = tmp_path / "stories.txt"
    text = "  ab story\n<|endoftext|>\n\n다음 이야기<|endoftext|>ab tail"
    source.write_text(text, encoding="utf-8")
    dataset = load_training_dataset(source, tokenizer, context_length=7, cache_dir=tmp_path / "cache")
    assert isinstance(dataset, Dataset)
    assert isinstance(dataset.data, MemoryMappedTable)
    assert dataset.features["input_ids"].feature.dtype == "uint16"
    assert dataset[0]["input_ids"].dtype == torch.long
    expected = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    assert len(dataset) == (len(expected) - 1) // 7
    for index, row in enumerate(dataset):
        assert row["input_ids"].tolist() == expected[index * 7:index * 7 + 8]
    rows = []
    for batch in DataLoader(dataset, batch_size=3, shuffle=True):
        tokens = batch["input_ids"]
        x, y = tokens[:, :-1], tokens[:, 1:]
        assert torch.equal(x[:, 1:], y[:, :-1])
        rows.extend(tokens.tolist())
    assert sorted(rows) == sorted(row["input_ids"].tolist() for row in dataset)


def test_cache_reused_and_invalidated_on_source_change(tmp_path, tokenizer):
    source = tmp_path / "stories.txt"
    source.write_text("ab story<|endoftext|>another story", encoding="utf-8")
    kwargs = dict(context_length=4, cache_dir=tmp_path / "cache")
    first = load_training_dataset(source, tokenizer, **kwargs)
    files = first.cache_files
    timestamps = [os.stat(entry["filename"]).st_mtime_ns for entry in files]
    second = load_training_dataset(source, tokenizer, **kwargs)
    assert first._fingerprint == second._fingerprint
    assert second.cache_files == files
    assert [os.stat(entry["filename"]).st_mtime_ns for entry in files] == timestamps
    source.write_text("a much longer and different story<|endoftext|>", encoding="utf-8")
    changed = load_training_dataset(source, tokenizer, **kwargs)
    assert changed.cache_files != files
    assert changed[0]["input_ids"].tolist() != first[0]["input_ids"].tolist()
    resized = load_training_dataset(source, tokenizer, context_length=8, cache_dir=tmp_path / "cache")
    assert resized.cache_files != changed.cache_files
    assert len(resized[0]["input_ids"]) == 9


def test_invalid_context_length(tmp_path, tokenizer):
    with pytest.raises(ValueError, match="positive"):
        load_training_dataset(tmp_path / "missing.txt", tokenizer, context_length=0)


def test_arrow_training_smoke(tmp_path, tokenizer, monkeypatch):
    from cs336_basics import train_model

    source = tmp_path / "stories.txt"
    source.write_text("ab story<|endoftext|>" * 10, encoding="utf-8")
    dataset = load_training_dataset(source, tokenizer, context_length=8, cache_dir=tmp_path / "cache")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, _, checkpoint = train_model.train(dataset.select(range(3)), tokenizer.n_vocab, num_epoch=2)
    assert checkpoint["global_step"] == 2
    assert checkpoint["epoch"] == 2
    assert checkpoint["model_config"]["max_kv_len"] == 8
    assert (tmp_path / "checkpoint.pt").exists()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
