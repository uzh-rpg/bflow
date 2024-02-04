from typing import List, Optional, Union, Any, Tuple

import numpy as np
import skimage
from skimage import img_as_ubyte
import torch
from torchvision.transforms import ColorJitter

from utils.general import inputs_to_tensor, wrap_unwrap_lists_for_class_method

UAT = Union[np.ndarray, torch.Tensor]
ULISTT = Union[List[torch.Tensor], torch.Tensor]
ULISTAT = Union[List[UAT], UAT]

def torch_img_to_numpy(torch_img: torch.Tensor):
    ch, ht, wd = torch_img.shape
    assert ch == 3
    numpy_img = torch_img.numpy()
    numpy_img = np.moveaxis(numpy_img, 0, -1)
    return numpy_img

def numpy_img_to_torch(numpy_img: np.ndarray):
    ht, wd, ch = numpy_img.shape
    assert ch == 3
    numpy_img = np.moveaxis(numpy_img, -1, 0)
    torch_img = torch.from_numpy(numpy_img)
    return torch_img

class PhotoAugmentor:
    def __init__(self,
                 brightness: float,
                 contrast: float,
                 saturation: float,
                 hue: float,
                 probability_color: float,
                 noise_variance_range: Tuple[float, float],
                 probability_noise: float):
        assert 0 <= probability_color <= 1
        assert 0 <= probability_noise <= 1
        assert len(noise_variance_range) == 2
        self.photo_augm = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.probability_color = probability_color
        self.probability_noise = probability_noise
        self.var_min = noise_variance_range[0]
        self.var_max = noise_variance_range[1]
        assert self.var_max > self.var_min

        self.seed = torch.randint(low=0, high=2**32, size=(1,))[0].item()

    @staticmethod
    def sample_uniform(min_value: float=0, max_value: float=1) -> float:
        assert max_value > min_value
        uni_sample = torch.rand(1)[0].item()
        return (max_value - min_value)*uni_sample + min_value

    @wrap_unwrap_lists_for_class_method
    def _apply_jitter(self, images: ULISTT):
        assert isinstance(images, list)

        for idx, entry in enumerate(images):
            images[idx] = self.photo_augm(entry)

        return images

    @wrap_unwrap_lists_for_class_method
    def _apply_noise(self, images: ULISTT):
        assert isinstance(images, list)
        variance = self.sample_uniform(min_value=0.001, max_value=0.01)

        for idx, entry in enumerate(images):
            assert isinstance(entry, torch.Tensor)
            numpy_img = torch_img_to_numpy(entry)
            noisy_img = skimage.util.random_noise(numpy_img, mode='speckle', var=variance, clip=True, seed=self.seed) # return float64 in [0, 1]
            noisy_img = img_as_ubyte(noisy_img)
            torch_img = numpy_img_to_torch(noisy_img)
            images[idx] = torch_img

        return images

    @inputs_to_tensor
    def __call__(self, images: ULISTAT):
        if self.probability_color > torch.rand(1).item():
            images = self._apply_jitter(images)
        if self.probability_noise > torch.rand(1).item():
            images = self._apply_noise(images)
        return images

class FlowAugmentor:
    def __init__(self,
                 crop_size_hw,
                 h_flip_prob: float=0.5,
                 v_flip_prob: float=0.1):
        assert crop_size_hw[0] > 0
        assert crop_size_hw[1] > 0
        assert 0 <= h_flip_prob <= 1
        assert 0 <= v_flip_prob <= 1

        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.crop_size_hw = crop_size_hw

    @wrap_unwrap_lists_for_class_method
    def _random_cropping(self, ev_repr: Optional[ULISTT], flow: Optional[ULISTT], valid: Optional[ULISTT]=None, images: Optional[ULISTT]=None):
        if ev_repr is not None:
            assert isinstance(ev_repr, list)
            height, width = ev_repr[0].shape[-2:]
        elif images is not None:
            assert isinstance(images, list)
            height, width = images[0].shape[-2:]
        else:
            raise NotImplementedError

        y0 = torch.randint(0, height - self.crop_size_hw[0], (1,)).item()
        x0 = torch.randint(0, width - self.crop_size_hw[1], (1,)).item()

        if ev_repr is not None:
            assert isinstance(ev_repr, list)
            nbins = ev_repr[0].shape[-3]
            for idx, entry in enumerate(ev_repr):
                assert entry.shape[-3:] == (nbins, height, width), f'actual (nbins, h, w) = ({entry.shape[-3]}, {entry.shape[-2]}, {entry.shape[-1]}), expected (nbins, h, w) = ({nbins}, {height}, {width})'
                # NOTE: Elements of a range-based for loop do not directly modify the original list in Python!
                ev_repr[idx] = entry[..., y0:y0+self.crop_size_hw[0], x0:x0+self.crop_size_hw[1]]

        if flow is not None:
            assert isinstance(flow, list)
            for idx, entry in enumerate(flow):
                assert entry.shape[-3:] == (2, height, width), f'actual (c, h, w) = ({entry.shape[-3]}, {entry.shape[-2]}, {entry.shape[-1]}), expected (2, h, w) = (2, {height}, {width})'
                flow[idx] = entry[..., y0:y0+self.crop_size_hw[0], x0:x0+self.crop_size_hw[1]]

        if valid is not None:
            assert isinstance(valid, list)
            for idx, entry in enumerate(valid):
                assert entry.shape[-2:] == (height, width), f'actual (h, w) = ({entry.shape[-2]}, {entry.shape[-1]}), expected (h, w) = ({height}, {width})'
                valid[idx] = entry[y0:y0+self.crop_size_hw[0], x0:x0+self.crop_size_hw[1]]

        if images is not None:
            assert isinstance(images, list)
            for idx, entry in enumerate(images):
                assert entry.shape[-2:] == (height, width), f'actual (h, w) = ({entry.shape[-2]}, {entry.shape[-1]}), expected (h, w) = ({height}, {width})'
                images[idx] = entry[..., y0:y0+self.crop_size_hw[0], x0:x0+self.crop_size_hw[1]]

        return ev_repr, flow, valid, images

    @wrap_unwrap_lists_for_class_method
    def _horizontal_flipping(self, ev_repr: Optional[ULISTT], flow: Optional[ULISTT], valid: Optional[ULISTT]=None, images: Optional[ULISTT]=None):
        # flip last axis which is assumed to be width
        if ev_repr is not None:
            assert isinstance(ev_repr, list)
            ev_repr = [x.flip(-1) for x in ev_repr]
        if images is not None:
            assert isinstance(images, list)
            images = [x.flip(-1) for x in images]
        if valid is not None:
            assert isinstance(valid, list)
            valid = [x.flip(-1) for x in valid]
        if flow is not None:
            assert isinstance(flow, list)
            flow = [x.flip(-1) for x in flow]
            # also flip the sign of the x component of the flow
            for idx, entry in enumerate(flow):
                flow[idx][0] = -1 * entry[0]

        return ev_repr, flow, valid, images

    @wrap_unwrap_lists_for_class_method
    def _vertical_flipping(self, ev_repr: Optional[ULISTT], flow: Optional[ULISTT], valid: Optional[ULISTT]=None, images: Optional[ULISTT]=None):
        # flip second last axis which is assumed to be height
        if ev_repr is not None:
            assert isinstance(ev_repr, list)
            ev_repr = [x.flip(-2) for x in ev_repr]
        if images is not None:
            assert isinstance(images, list)
            images = [x.flip(-2) for x in images]
        if valid is not None:
            assert isinstance(valid, list)
            valid = [x.flip(-2) for x in valid]
        if flow is not None:
            assert isinstance(flow, list)
            flow = [x.flip(-2) for x in flow]
            # also flip the sign of the y component of the flow
            for idx, entry in enumerate(flow):
                flow[idx][1] = -1 * entry[1]

        return ev_repr, flow, valid, images

    @inputs_to_tensor
    def __call__(self,
                 ev_repr: Optional[ULISTAT]=None,
                 flow: Optional[ULISTAT]=None,
                 valid: Optional[ULISTAT]=None,
                 images: Optional[ULISTAT]=None):

        if self.h_flip_prob > torch.rand(1).item():
            ev_repr, flow, valid, images = self._horizontal_flipping(ev_repr, flow, valid, images)

        if self.v_flip_prob > torch.rand(1).item():
            ev_repr, flow, valid, images = self._vertical_flipping(ev_repr, flow, valid, images)

        ev_repr, flow, valid, images = self._random_cropping(ev_repr, flow, valid, images)

        out = (ev_repr, flow, valid, images)
        out = (x for x in out if x is not None)
        return out
