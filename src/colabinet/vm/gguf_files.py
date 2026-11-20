"""GGUF split filenames, shared by local size checks and the standalone VM installer."""

import re


def gguf_files(path: str) -> list[str]:
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path)
    if not match:
        return [path]
    prefix, index, count = match.groups()
    if int(index) != 1 or int(count) < 1:
        raise ValueError("select the first shard (00001) of a split GGUF model")
    return [f"{prefix}-{i:05d}-of-{count}.gguf" for i in range(1, int(count) + 1)]


def repo_files(path: str, links: dict | None = None) -> list[str]:
    """Files that really exist in the Hugging Face repo for `path`: every shard llama.cpp expects, except the
    names a preset's `links` alias to another file, plus those link targets (order kept, no duplicates).
    Use this, not gguf_files, wherever a shard list is fetched or HEADed from the Hub."""
    links = links or {}
    files = [name for name in gguf_files(path) if name not in links] + list(links.values())
    return list(dict.fromkeys(files))
