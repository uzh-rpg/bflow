from collections.abc import Mapping
from functools import wraps

import torch
import torch.nn.functional as F


def detach_tensors(cpu: bool=False):
    ''' Detach tensors and optionally move to cpu. Only detaches torch.Tensor instances '''
    # Decorator factory to enable decorator arguments: https://stackoverflow.com/a/50538967
    def allow_detach(key: str, value, train: bool):
        if train and key == 'loss':
            return False
        return isinstance(value, torch.Tensor)
    def detach_tensor(input_tensor: torch.Tensor, cpu: bool):
        assert isinstance(input_tensor, torch.Tensor)
        if cpu:
            return input_tensor.detach().cpu()
        return input_tensor.detach()
    def decorator(func):
        train = 'train' in func.__name__

        @wraps(func)
        def inner(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, Mapping):
                return {k: (detach_tensor(v, cpu) if allow_detach(k, v, train) else v) for k, v in output.items()}
            assert isinstance(output, torch.Tensor)
            if train:
                # Do not detach because this will be the loss function of the training hook, which must not be detached.
                return output
            return detach_tensor(output, cpu)
        return inner
    return decorator


def reduce_ev_repr(ev_repr: torch.Tensor) -> torch.Tensor:
    # This function is useful to reduce the overhead of moving an event representation
    # to CPU for visualization.
    # For now simply sum up the time dimension to reduce the memory.
    assert isinstance(ev_repr, torch.Tensor)
    assert ev_repr.ndim == 4
    assert ev_repr.is_cuda

    return torch.sum(ev_repr, dim=1)


class InputPadder:
    """ Pads input tensor such that the last two dimensions are divisible by min_size """
    def __init__(self, min_size: int=8, no_top_padding: bool=False):
        assert min_size > 0
        self.min_size = min_size
        self.no_top_padding = no_top_padding
        self._pad = None

    def requires_padding(self, input_tensor: torch.Tensor):
        ht, wd = input_tensor.shape[-2:]
        answer = False
        answer &= ht % self.min_size == 0
        answer &= wd % self.min_size == 0
        return answer

    def pad(self, input_tensor: torch.Tensor):
        ht, wd = input_tensor.shape[-2:]
        pad_ht = (((ht // self.min_size) + 1) * self.min_size - ht) % self.min_size
        pad_wd = (((wd // self.min_size) + 1) * self.min_size - wd) % self.min_size
        if self.no_top_padding:
            # Pad only bottom instead of top
            # RAFT uses this for KITTI
            pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        else:
            # RAFT uses this for SINTEL (as default too)
            pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        if self._pad is None:
            self._pad = pad
        else:
            assert self._pad == pad
        return F.pad(input_tensor, self._pad, mode='replicate')

    def unpad(self, input_tensor: torch.Tensor):
        ht, wd = input_tensor.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return input_tensor[..., c[0]:c[1], c[2]:c[3]]
