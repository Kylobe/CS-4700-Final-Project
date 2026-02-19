import os
import glob
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


Sample = Any  # can be (x, policy, value), dict, etc.

def _iter_samples_from_loaded(obj: Any) -> Iterator[Sample]:
    """
    Normalize whatever torch.load() returns into an iterator over samples.
    Supports:
      - list/tuple of samples
      - dict of batched tensors (same first dim) -> yields dict per row
      - single sample (dict/tensor/tuple/etc.)
    """
    # Case 1: list/tuple of samples already
    if isinstance(obj, (list, tuple)):
        for s in obj:
            yield s
        return

    # Case 2: dict of batched tensors
    if isinstance(obj, dict):
        # If all values are tensors with same first dimension -> treat as batch
        tensor_vals = [v for v in obj.values() if torch.is_tensor(v)]
        if tensor_vals and all(v.ndim >= 1 for v in tensor_vals):
            n = tensor_vals[0].shape[0]
            if all(v.shape[0] == n for v in tensor_vals):
                for i in range(n):
                    yield {k: (v[i] if torch.is_tensor(v) else v) for k, v in obj.items()}
                return

        # Otherwise, treat dict itself as a single sample
        yield obj
        return

    # Case 3: anything else is a single sample
    yield obj


class PtShardStream(IterableDataset):
    """
    Streams samples from many .pt shard files on disk.

    - Each worker gets a disjoint subset of files.
    - Optional shuffle buffer approximates global shuffling without loading everything.
    """

    def __init__(
        self,
        root_dir: str,
        pattern: str = "**/*.pt",
        seed: int = 0,
        shuffle_files_each_epoch: bool = True,
        shuffle_buffer: int = 0,  # 0 disables streaming shuffle
        map_location: str = "cpu",
        drop_errors: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.pattern = pattern
        self.seed = seed
        self.shuffle_files_each_epoch = shuffle_files_each_epoch
        self.shuffle_buffer = int(shuffle_buffer)
        self.map_location = map_location
        self.drop_errors = drop_errors

        self._all_files = sorted(glob.glob(os.path.join(root_dir, pattern), recursive=True))
        if not self._all_files:
            raise FileNotFoundError(f"No .pt files found under {root_dir} with pattern '{pattern}'")

        # Used to vary file order per epoch without needing __len__
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call this from your training loop each epoch to reshuffle file order deterministically."""
        self._epoch = int(epoch)

    def _get_worker_files(self) -> List[str]:
        files = self._all_files

        # Shuffle file order per epoch (same across workers, then worker-split)
        if self.shuffle_files_each_epoch:
            rng = random.Random(self.seed + self._epoch)
            files = files.copy()
            rng.shuffle(files)

        info = get_worker_info()
        if info is None:
            return files

        # Split files across workers: worker i gets files[i::num_workers]
        return files[info.id :: info.num_workers]

    def __iter__(self) -> Iterator[Sample]:
        worker_files = self._get_worker_files()

        # For per-worker RNG (streaming shuffle buffer)
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        rng = random.Random(self.seed + 10_000 * self._epoch + worker_id)

        buffer: List[Sample] = []
        use_buffer = self.shuffle_buffer > 0

        def emit_from_buffer() -> Iterator[Sample]:
            # pop a random element from buffer
            j = rng.randrange(len(buffer))
            buffer[j], buffer[-1] = buffer[-1], buffer[j]
            return (buffer.pop(),)

        for fp in worker_files:
            try:
                # NOTE: .pt shards should be saved in a way that doesn't force GPU tensors on load.
                obj = torch.load(fp, map_location=self.map_location, weights_only=False)
            except Exception as e:
                if self.drop_errors:
                    continue
                raise RuntimeError(f"Failed to torch.load shard: {fp}") from e

            for sample in _iter_samples_from_loaded(obj):
                if not use_buffer:
                    yield sample
                    continue

                buffer.append(sample)
                if len(buffer) >= self.shuffle_buffer:
                    # emit one randomized sample each time we add beyond threshold
                    yield from emit_from_buffer()

        # Flush remaining buffer at end
        if use_buffer:
            while buffer:
                yield from emit_from_buffer()

def _as_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        # If you ever hit "not writable" warnings, swap to: np.array(x, copy=True)
        return torch.from_numpy(x)
    # python numbers / lists / tuples
    return torch.as_tensor(x)

def smart_collate(batch: Sequence[Any]) -> Any:
    first = batch[0]

    if isinstance(first, dict):
        out: Dict[str, Any] = {}
        for k in first.keys():
            vals = [b[k] for b in batch]

            # stack if *all* are stackable tensor-ish, otherwise keep list
            if all(torch.is_tensor(v) or isinstance(v, (np.ndarray, int, float, list, tuple)) for v in vals):
                # only stack if shapes match (otherwise you probably want a list)
                try:
                    tvals = [_as_tensor(v) for v in vals]
                    out[k] = torch.stack(tvals, dim=0)
                except Exception:
                    out[k] = vals
            else:
                out[k] = vals
        return out

    if isinstance(first, (tuple, list)):
        transposed = list(zip(*batch))
        stacked = []
        for vals in transposed:
            vals = list(vals)

            # try to stack after converting numpy/scalars -> tensor
            try:
                tvals = [_as_tensor(v) for v in vals]
                # only stack if all tensors have same shape
                if all(t.shape == tvals[0].shape for t in tvals):
                    stacked.append(torch.stack(tvals, dim=0))
                else:
                    stacked.append(vals)  # variable-length things (like move lists)
            except Exception:
                stacked.append(vals)

        return tuple(stacked) if isinstance(first, tuple) else stacked

    # single item per sample
    try:
        tvals = [_as_tensor(v) for v in batch]
        if all(t.shape == tvals[0].shape for t in tvals):
            return torch.stack(tvals, dim=0)
    except Exception:
        pass

    return list(batch)

def make_stream_loader(
    root_dir: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle_buffer: int = 50_000,  # adjust based on sample size/RAM
    seed: int = 0,
    pin_memory: bool = True,       # if using GPU
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[PtShardStream, DataLoader]:
    ds = PtShardStream(
        root_dir=root_dir,
        pattern="**/*.pt",
        seed=seed,
        shuffle_files_each_epoch=True,
        shuffle_buffer=shuffle_buffer,
        map_location="cpu",
        drop_errors=False,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=smart_collate,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    )

    return ds, loader






