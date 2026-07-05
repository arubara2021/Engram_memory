from __future__ import annotations

from typing import List


def fnv1a(text: str, table_size: int) -> int:
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % table_size


def murmur3(text: str, table_size: int) -> int:
    h = 0x9747B28C
    data = text.encode("utf-8")
    for b in data:
        h ^= b
        h = (h * 0x5BD1E995) & 0xFFFFFFFF
        h ^= (h >> 15)
    h ^= len(data)
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= (h >> 13)
    return h % table_size


def djb2(text: str, table_size: int) -> int:
    h = 5381
    for b in text.encode("utf-8"):
        h = ((h << 5) + h + b) & 0xFFFFFFFF
    return h % table_size


_HASH_FUNCTIONS = [fnv1a, murmur3, djb2]


def hash_all(text: str, table_size: int) -> List[int]:
    return [fn(text, table_size) for fn in _HASH_FUNCTIONS]


def fnv1a_raw(data: bytes, table_size: int) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % table_size


def murmur3_raw(data: bytes, table_size: int) -> int:
    h = 0x9747B28C
    for b in data:
        h ^= b
        h = (h * 0x5BD1E995) & 0xFFFFFFFF
        h ^= (h >> 15)
    h ^= len(data)
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= (h >> 13)
    return h % table_size


def djb2_raw(data: bytes, table_size: int) -> int:
    h = 5381
    for b in data:
        h = ((h << 5) + h + b) & 0xFFFFFFFF
    return h % table_size