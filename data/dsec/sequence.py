from pathlib import Path
from typing import Type

import numpy as np
import torch

from data.dsec.subsequence.base import BaseSubSequence


class SubSequenceGenerator:
    # seq_name (e.g. zurich_city_10_a)
    # ├── flow
    # │   ├── backward (for train only)
    # │   │   ├── xxxxxx.png
    # │   │   └── ...
    # │   ├── backward_timestamps.txt (for train only)
    # │   ├── forward (for train only)
    # │   │   ├── xxxxxx.png
    # │   │   └── ...
    # │   └── forward_timestamps.txt (for train and test)
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5
    def __init__(self,
        seq_path: Path,
        subseq_class: Type[BaseSubSequence],
        args: dict):

        self.args = args
        self.seq_path = seq_path

        self.seqseq_class = subseq_class

        # load forward optical flow timestamps
        assert sequence_has_flow(seq_path), seq_path
        flow_dir = seq_path / 'flow'
        assert flow_dir.is_dir(), str(flow_dir)
        forward_timestamps_file = flow_dir / 'forward_timestamps.txt'
        assert forward_timestamps_file.is_file()
        self.forward_flow_timestamps = np.loadtxt(str(forward_timestamps_file), dtype='int64', delimiter=',')
        assert self.forward_flow_timestamps.ndim == 2
        assert self.forward_flow_timestamps.shape[1] == 2

        # load forward optical flow paths
        forward_flow_dir = flow_dir / 'forward'
        assert forward_flow_dir.is_dir()
        forward_flow_list = list()
        for entry in forward_flow_dir.iterdir():
            assert str(entry.name).endswith('.png')
            forward_flow_list.append(entry)
        forward_flow_list.sort()
        self.forward_flow_list = forward_flow_list

        # Extract start indices of sub-sequences
        from_ts = self.forward_flow_timestamps[:, 0]
        to_ts = self.forward_flow_timestamps[:, 1]

        is_start_subseq = from_ts[1:] != to_ts[:-1]
        # Add first index as start index too.
        is_start_subseq = np.concatenate((np.array((True,), dtype='bool'), is_start_subseq))
        self.start_indices = list(np.where(is_start_subseq)[0])

        self.subseq_idx = 0

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.start_indices)

    def __next__(self) -> BaseSubSequence:
        if self.subseq_idx >= len(self.start_indices):
            raise StopIteration
        final_subseq = self.subseq_idx == len(self.start_indices) - 1

        start_idx = self.start_indices[self.subseq_idx]
        end_p1_idx = None if final_subseq else self.start_indices[self.subseq_idx + 1]

        forward_flow_timestamps = self.forward_flow_timestamps[start_idx:end_p1_idx, :]
        forward_flow_list = self.forward_flow_list[start_idx:end_p1_idx]

        self.subseq_idx += 1

        return self.seqseq_class(self.seq_path, forward_flow_timestamps, forward_flow_list, **self.args)


def sequence_has_flow(seq_path: Path):
    return (seq_path / 'flow').is_dir()


def generate_sequence(
        seq_path: Path,
        subseq_class: Type[BaseSubSequence],
        args: dict):
    if not sequence_has_flow(seq_path):
        return None

    subseq_list = list()

    for subseq in SubSequenceGenerator(seq_path, subseq_class, args):
        subseq_list.append(subseq)

    return torch.utils.data.ConcatDataset(subseq_list)
