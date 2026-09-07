"""Length-bucket batch sampler deterministic cho long-context training."""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence


DEFAULT_BUCKET_BOUNDARIES = (2048, 4096, 6144, 8192)


class LengthBucketBatchSampler:
    """Bucket trước, sau đó shard các global micro-batch cho DDP.

    ``batch_size`` là local batch size. Một global micro-batch có
    ``batch_size * world_size`` sample rồi được chia đều cho các rank; vì vậy
    gradient accumulation vẫn giữ đúng semantics của Trainer hiện tại.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        boundaries: Sequence[int] = DEFAULT_BUCKET_BOUNDARIES,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("batch_size/rank/world_size không hợp lệ")
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.boundaries = tuple(int(x) for x in boundaries)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _bucket_id(self, length: int) -> int:
        for index, boundary in enumerate(self.boundaries):
            if length <= boundary:
                return index
        return len(self.boundaries)

    def _all_rank_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets: List[List[int]] = [[] for _ in range(len(self.boundaries) + 1)]
        for index in range(len(self.lengths)):
            buckets[self._bucket_id(self.lengths[index])].append(index)
        global_size = self.batch_size * self.world_size
        global_batches: List[List[int]] = []
        for bucket in buckets:
            rng.shuffle(bucket)
            usable = (len(bucket) // global_size) * global_size
            if not self.drop_last and usable < len(bucket):
                usable = len(bucket)
            for start in range(0, usable, global_size):
                batch = bucket[start : start + global_size]
                if len(batch) < global_size and self.drop_last:
                    continue
                global_batches.append(batch)
        rng.shuffle(global_batches)
        return [
            batch[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]
            for batch in global_batches
            if len(batch) >= global_size
        ]

    def __iter__(self) -> Iterable[List[int]]:
        yield from self._all_rank_batches()

    def __len__(self) -> int:
        global_size = self.batch_size * self.world_size
        return sum(len(bucket) // global_size for bucket in self._bucketed_lengths())

    def _bucketed_lengths(self) -> List[List[int]]:
        buckets: List[List[int]] = [[] for _ in range(len(self.boundaries) + 1)]
        for length in self.lengths:
            buckets[self._bucket_id(length)].append(length)
        return buckets


__all__ = ["DEFAULT_BUCKET_BOUNDARIES", "LengthBucketBatchSampler"]
