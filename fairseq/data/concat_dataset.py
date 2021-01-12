# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import logging
import numpy as np
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset


class ConcatDataset(FairseqDataset):
    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1, homogeneous_batch=False):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]
        self.homogeneous_batch = homogeneous_batch

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if hasattr(self.datasets[0], "collater"):
            return self.datasets[0].collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def sizes(self):
        _dataset_sizes = []
        for ds, sr in zip(self.datasets, self.sample_ratios):
            if isinstance(ds.sizes, np.ndarray):
                _dataset_sizes.append(np.tile(ds.sizes, sr))
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.sizes, list)
                _dataset_sizes.append(np.tile(ds.sizes[0], sr))
        return np.concatenate(_dataset_sizes)

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

    def ordered_indices(self):
        """
        Returns indices sorted by length. So less padding is needed.
        """
        if not self.homogeneous_batch:
            if isinstance(self.sizes, np.ndarray) and len(self.sizes.shape) > 1:
                # special handling for concatenating lang_pair_datasets
                indices = np.arange(len(self))
                sizes = self.sizes
                tgt_sizes = (
                    sizes[:, 1] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else None
                )
                src_sizes = (
                    sizes[:, 0] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else sizes
                )
                # sort by target length, then source length
                if tgt_sizes is not None:
                    indices = indices[np.argsort(tgt_sizes[indices], kind="mergesort")]
                return indices[np.argsort(src_sizes[indices], kind="mergesort")]
            else:
                return np.argsort(self.sizes)
        else:
            # Sort for each subset
            sorted_indices = np.zeros(len(self.sizes), dtype=np.int64)
            start_idx = 0
            for d in self.datasets:
                stop_idx = start_idx + len(d)
                sorted_indices[start_idx:stop_idx] = np.argsort(self.sizes[start_idx:stop_idx]) + start_idx
                start_idx += len(d)
            
            return sorted_indices

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        if not self.homogeneous_batch:
            return super().batch_by_size(
                    indices=indices,
                    max_tokens=max_tokens,
                    max_sentences=max_sentences,
                    required_batch_size_multiple=required_batch_size_multiple,
                    )
        else:
        # If we want homogeneous batch, i.e., each batch contains only samples
        # drawn from the same subset
            return self._batch_by_size_homogeneous(
                    indices=indices,
                    max_tokens=max_tokens,
                    max_sentences=max_sentences,
                    required_batch_size_multiple=required_batch_size_multiple,
                    )
            

    def _batch_by_size_homogeneous(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        start_idx = 0
        batch_samplers = [None]*len(self.datasets)
        num_samples= [len(s) for s in self.datasets]
        # logging.info(f'Number of samples per datasets: {num_samples}')
        for i, d in enumerate(self.datasets):
            # logging.info(f'dataset {i}: start_idx: {start_idx}')
            # logging.info(f'dataset {i} indices: {indices[start_idx:start_idx+len(d)]}')
            batch_samplers[i] = super().batch_by_size(
                        indices=indices[start_idx : start_idx+len(d)],
                        max_tokens=max_tokens,
                        max_sentences=max_sentences,
                        required_batch_size_multiple=required_batch_size_multiple,
                        )
            start_idx += len(d)
        # Create a new batch sampler that choose randomly a batch among the above
        # batch samplers
        # iterators = list(map(iter, batch_samplers))
        # while iterators:         
        #     iterator = np.random.choice(iterators)
        #     try:
        #         yield next(iterator)
        #     except StopIteration:
        #         iterators.remove(iterator)
        # iterators = [iter(s) for s in batch_samplers]
        iterators = list(map(iter, batch_samplers))
        # num_batches = [len(s) for s in batch_samplers]
        # logging.info(f'Number of iterators: {len(iterators)}')
        # logging.info(f'Number of batches per datasets: {num_batches}')
        while iterators:  
            for i, iterator in enumerate(iterators):
                try:
                    yield next(iterator)
                except StopIteration:  
                    iterators.remove(iterator)

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cumulative_sizes, self.datasets):
            real_size = len(ds)
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(d.can_reuse_epoch_itr_across_epochs for d in self.datasets)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
