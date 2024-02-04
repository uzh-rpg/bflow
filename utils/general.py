import numbers
from functools import wraps
from typing import Any, Dict, Union, List

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


def is_cpu(input_: Union[torch.Tensor, List[torch.Tensor]]) -> bool:
    if isinstance(input_, torch.Tensor):
        return input_.device == torch.device('cpu')
    assert isinstance(input_, list)
    on_cpu = True
    for x in input_:
        assert isinstance(x, torch.Tensor)
        on_cpu &= x.device == torch.device('cpu')
    return on_cpu


def _convert_to_tensor(input_: Any):
    if input_ is None or isinstance(input_, torch.Tensor):
        return input_
    if isinstance(input_, np.ndarray):
        return torch.from_numpy(input_)
    if isinstance(input_, numbers.Number):
        return torch.tensor(input_)
    if isinstance(input_, dict):
        return {k: _convert_to_tensor(v) for k, v in input_.items()}
    if isinstance(input_, list):
        return [_convert_to_tensor(x) for x in input_]
    if isinstance(input_, tuple):
        return (_convert_to_tensor(x) for x in input_)
    return input_


def inputs_to_tensor(func):
    @wraps(func)
    def inner(*args, **kwargs):
        args = _convert_to_tensor(args)
        kwargs = _convert_to_tensor(kwargs)
        return func(*args, **kwargs)

    return inner


def _obj_has_function(obj, func_name: str):
    return hasattr(obj, func_name) and callable(getattr(obj, func_name))


def _data_to_cpu(input_: Any):
    if input_ is None:
        return input_
    if isinstance(input_, torch.Tensor):
        return input_.cpu()
    if isinstance(input_, dict):
        return {k: _data_to_cpu(v) for k, v in input_.items()}
    if isinstance(input_, list):
        return [_data_to_cpu(x) for x in input_]
    if isinstance(input_, tuple):
        return (_data_to_cpu(x) for x in input_)
    assert _obj_has_function(input_, 'cpu')
    return input_.cpu()


def to_cpu(func):
    ''' Move stuff to cpu '''

    @wraps(func)
    def inner(*args, **kwargs):
        output = func(*args, **kwargs)
        output = _data_to_cpu(output)
        return output

    return inner


def _unwrap_len1_list_or_tuple(input_: Any):
    if isinstance(input_, tuple):
        if len(input_) == 1:
            return input_[0]
        return (_unwrap_len1_list_or_tuple(x) for x in input_)
    if isinstance(input_, list):
        if len(input_) == 1:
            return input_[0]
        return [_unwrap_len1_list_or_tuple(x) for x in input_]
    if isinstance(input_, dict):
        return {k: _unwrap_len1_list_or_tuple(v) for k, v in input_.items()}
    return input_


def wrap_unwrap_lists_for_class_method(func):
    # We add "self" such that it can (only) be used on class methods
    @wraps(func)
    def inner(self, *args, **kwargs):
        # The reason why we have to explicitly add self in the arguments is that
        # we wrap the inputs in a list and would need to detect whether the input arg is self or not.
        args = [arg if isinstance(arg, list) or arg is None else [arg] for arg in args]
        kwargs = {k: (v if isinstance(v, list) or v is None else [v]) for k, v in kwargs.items()}
        out = func(self, *args, **kwargs)
        out = _unwrap_len1_list_or_tuple(out)
        return out

    return inner


def get_ckpt_callback(config: Dict) -> ModelCheckpoint:
    dataset_name = config['dataset']['name']
    assert dataset_name in {'dsec', 'multiflow_regen'}

    if dataset_name == 'dsec':
        ckpt_callback_monitor = 'global_step'
        ckpt_filename = 'epoch={epoch:03d}-step={' + ckpt_callback_monitor + ':.0f}'
        mode = 'max'  # because of global_step
    else:
        prefix = 'val'
        metric = 'epe_multi'
        ckpt_callback_monitor = f'{prefix}/{metric}'
        filename_monitor_string = f'{prefix}_{metric}'
        ckpt_filename = 'epoch={epoch:03d}-step={step}-' + filename_monitor_string + '={' + ckpt_callback_monitor + ':.2f}'
        mode = 'min'

    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_callback_monitor,
        filename=ckpt_filename,
        auto_insert_metric_name=False,
        save_top_k=1,
        mode=mode,
        every_n_epochs=config['logging']['ckpt_every_n_epochs'],
        save_last=True,
        verbose=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = 'last_epoch={epoch:03d}-step={step}'
    return checkpoint_callback