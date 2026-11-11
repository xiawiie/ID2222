"""Homework 1 的核心实现。

这个文件只放可以重复使用的代码：

* 读取 Twenty Newsgroups 文档
* Shingling
* Jaccard similarity
* MinHashing
* Signature similarity
* LSH

命令行实验在 main.py，Jupyter 分析文档会导入本文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    """一篇文档及其来源信息。"""

    name: str
    category: str
    text: str
    path: Path


def preprocess_text(raw_text: str, include_subject: bool = True) -> str:
    """清洗一篇 Twenty Newsgroups 消息。

    数据文件看起来像邮件，第一段是 headers，空行后面是正文。
    Subject 对判断新闻主题有帮助，因此默认保留 Subject 和正文。
    Message-ID、日期等字段含有大量文档特有的噪声，所以不保留。

    如果 include_subject 为 False，则只使用正文。这个参数用于 Notebook
    中对比预处理方式，而不是为了增加项目复杂度。
    """

    parts = re.split(r"\r?\n\r?\n", raw_text, maxsplit=1)
    header_text = parts[0] if parts else ""
    body = parts[1] if len(parts) == 2 else raw_text

    subject = ""
    for line in header_text.splitlines():
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            break

    selected_text = f"{subject}\n{body}" if include_subject else body
    selected_text = selected_text.lower()
    return re.sub(r"\s+", " ", selected_text).strip()


class Shingling:
    """把文档转换成去重后的 shingle 哈希值集合。"""

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k

    @staticmethod
    def hash_shingle(shingle: str) -> int:
        """使用稳定哈希值表示一个 shingle。

        Python 内置 hash 在不同进程中可能随机变化，因此使用 SHA-1。
        这里只取前 8 个字节作为整数，足以作为本作业的 shingle 表示。
        """

        digest = hashlib.sha1(shingle.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def shingles(self, text: str) -> list[int]:
        """返回文档中所有唯一的 k-shingle 哈希值。"""

        if len(text) < self.k:
            return []

        values = {
            self.hash_shingle(text[index : index + self.k])
            for index in range(len(text) - self.k + 1)
        }
        # 排序不是算法必须的，但它让调试输出和测试结果稳定、易读。
        return sorted(values)


class CompareSets:
    """计算两个 shingle 集合的 Jaccard similarity。"""

    @staticmethod
    def jaccard(first: Iterable[int], second: Iterable[int]) -> float:
        first_set = set(first)
        second_set = set(second)
        union_size = len(first_set | second_set)
        if union_size == 0:
            # 两篇空文档在本项目中视为完全相同。
            return 1.0
        return len(first_set & second_set) / union_size


class MinHashing:
    """使用确定性的线性哈希函数生成 MinHash signature。"""

    # 大素数，用于线性哈希函数的模运算。
    PRIME = 2_305_843_009_213_693_951

    def __init__(self, signature_length: int = 100, seed: int = 42) -> None:
        if signature_length <= 0:
            raise ValueError("signature_length must be positive")

        self.signature_length = signature_length
        generator = random.Random(seed)
        self.coefficients = [
            (generator.randrange(1, self.PRIME), generator.randrange(self.PRIME))
            for _ in range(signature_length)
        ]

    def signature(self, hashed_shingles: Iterable[int]) -> list[int]:
        """将一个 shingle 集合转换成固定长度的整数向量。"""

        values = list(hashed_shingles)
        if not values:
            # 空集合没有真正的最小哈希值，用 PRIME 作为明确的占位值。
            return [self.PRIME] * self.signature_length

        signature: list[int] = []
        for multiplier, offset in self.coefficients:
            minimum = min(
                (multiplier * (value % self.PRIME) + offset) % self.PRIME
                for value in values
            )
            signature.append(minimum)
        return signature


class CompareSignatures:
    """用两个 MinHash signature 相同位置的比例估计 Jaccard。"""

    @staticmethod
    def similarity(first: list[int], second: list[int]) -> float:
        if len(first) != len(second):
            raise ValueError("signatures must have the same length")
        if not first:
            return 1.0
        equal_positions = sum(left == right for left, right in zip(first, second))
        return equal_positions / len(first)


class LSH:
    """使用 banding 找出可能相似的 signature 对。"""

    @staticmethod
    def candidates(
        signatures: dict[str, list[int]], bands: int = 20, rows: int = 5
    ) -> set[tuple[str, str]]:
        """返回 LSH 候选文档对。

        bands * rows 必须等于 signature 长度。这样每个 signature 的所有
        components 都被使用，不会静默丢掉尾部。
        """

        if not signatures:
            return set()
        if bands <= 0 or rows <= 0:
            raise ValueError("bands and rows must be positive")

        signature_lengths = {len(signature) for signature in signatures.values()}
        if len(signature_lengths) != 1:
            raise ValueError("all signatures must have the same length")
        signature_length = signature_lengths.pop()
        if bands * rows != signature_length:
            raise ValueError("bands * rows must equal signature length")

        candidates: set[tuple[str, str]] = set()
        names = sorted(signatures)
        for band in range(bands):
            start = band * rows
            end = start + rows
            buckets: dict[tuple[int, ...], list[str]] = {}
            for name in names:
                key = tuple(signatures[name][start:end])
                buckets.setdefault(key, []).append(name)

            for bucket_names in buckets.values():
                for first, second in combinations(sorted(bucket_names), 2):
                    candidates.add((first, second))
        return candidates


def read_document(path: Path, include_subject: bool = True) -> str:
    """以 Latin-1 读取一篇数据文件并执行预处理。"""

    raw_text = path.read_text(encoding="latin-1", errors="replace")
    return preprocess_text(raw_text, include_subject=include_subject)


def load_documents(
    data_dir: Path,
    categories: list[str] | None = None,
    include_subject: bool = True,
) -> list[Document]:
    """读取 Twenty Newsgroups 目录中的文档。

    categories 为 None 时读取全部类别，否则只读取指定类别。
    """

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    selected_categories = set(categories) if categories else None
    documents: list[Document] = []
    category_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    for category_dir in category_dirs:
        if selected_categories is not None and category_dir.name not in selected_categories:
            continue
        for file_path in sorted(path for path in category_dir.iterdir() if path.is_file()):
            documents.append(
                Document(
                    name=f"{category_dir.name}/{file_path.name}",
                    category=category_dir.name,
                    text=read_document(file_path, include_subject=include_subject),
                    path=file_path,
                )
            )
    return documents


def balanced_sample(documents: list[Document], size: int) -> list[Document]:
    """从不同类别轮流取文档，形成确定性的平衡样本。"""

    if size <= 0:
        raise ValueError("size must be positive")
    if size > len(documents):
        raise ValueError("size cannot be larger than the number of documents")

    groups: dict[str, list[Document]] = {}
    for document in documents:
        groups.setdefault(document.category, []).append(document)

    result: list[Document] = []
    category_names = sorted(groups)
    positions = {category: 0 for category in category_names}
    index = 0
    while len(result) < size and category_names:
        category = category_names[index % len(category_names)]
        position = positions[category]
        group = groups[category]
        if position < len(group):
            result.append(group[position])
            positions[category] += 1
        else:
            category_names.remove(category)
            index -= 1
        index += 1
    return result
