import io
import math
from typing import List, Optional, Union

import cv2

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("agg")

import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from torchvision.utils import make_grid

from models.raft_spline.bezier import BezierCurves
from utils.general import is_cpu

from callbacks.utils.flow_vis import flow_to_color

def flow_to_bgr_previous(flow_map: np.ndarray, gamma: float=0.4):
    assert flow_map.ndim == 3, flow_map.ndim
    h, w, c = flow_map.shape
    assert c == 2, c
    # assumes (H, W, 2)
    # from https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
    # gamma correction
    mag = mag/mag.max()
    mag = np.power(mag, gamma)

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def flow_to_rgb_previous(flow_map: np.ndarray, gamma: float=0.4):
    bgr = flow_to_bgr_previous(flow_map, gamma)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def img_torch_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    assert img_tensor.ndim >= 3
    img_array = img_tensor.numpy()
    # *, c, h, w -> *, h, w, c
    img_array = np.moveaxis(img_array, -3, -1)
    return img_array

def batch_to_column(input_: torch.Tensor):
    if input_.ndim == 3:
        # b,h,w ->  b*h,w
        return torch.flatten(input_, start_dim=0, end_dim=1)
    assert input_.ndim == 4
    # b,c,h,w ->  c,b,h,w
    input_ = torch.transpose(input_, 0, 1)
    # c,b,h,w ->  c,b*h,w
    return torch.flatten(input_, start_dim=1, end_dim=2)

def flow_torch2numpy(flow: torch.Tensor):
    assert is_cpu(flow)
    flow = flow.numpy()
    flow = np.moveaxis(flow, 0, -1) # c,h,w -> h,w,c
    # The network can also return float16 optionally.
    return flow.astype('float32')

def flow_to_img(flow: torch.Tensor, use_mpi_style: bool=True):
    flow = flow_torch2numpy(flow)
    if use_mpi_style:
        rgb = flow_to_color(flow)
        return rgb
    rgb = flow_to_rgb_previous(flow)
    return rgb

def ev_repr_reduced_to_img_grayscale(ev_repr_reduced: torch.Tensor):
    assert is_cpu(ev_repr_reduced)
    ev_repr_reduced = ev_repr_reduced.numpy()
    ev_repr_reduced: np.ndarray
    img_1c = (ev_repr_reduced/ev_repr_reduced.max()*256).astype('uint8')
    img_3c = np.stack((img_1c,)*3, axis=-1)
    return img_3c

def ev_repr_reduced_to_img(ev_repr_reduced: torch.Tensor):
    assert is_cpu(ev_repr_reduced)

    ev_in = ev_repr_reduced.numpy()
    ev_in: np.ndarray
    ht, wd = ev_in.shape[-2:]

    out = np.full((ht, wd, 3), fill_value=255, dtype='uint8')
    #out[ev_in > 0] = [0, 0, 255]
    #out[ev_in < 0] = [255, 0, 0]
    #out = out.astype('uint8')

    clip_percentile = 50
    min_percentile = -np.percentile(np.abs(ev_in[ev_in < 0]), clip_percentile)
    max_percentile = np.percentile(np.abs(ev_in[ev_in > 0]), clip_percentile)
    ev_in = np.clip(ev_in, min_percentile, max_percentile)

    scaling = 2

    ev_in_max = ev_in.max()
    idx_pos = ev_in > 0
    ev_in[idx_pos] = (ev_in[idx_pos]/ev_in_max)**scaling
    val_pos = ev_in[idx_pos]
    out[idx_pos] = np.stack((255-val_pos*255, 255-val_pos*255, np.ones_like(val_pos)*255), axis=1)

    ev_in_min = ev_in.min()
    idx_neg = ev_in < 0
    ev_in[idx_neg] = (ev_in[idx_neg]/ev_in_min)**scaling
    val_neg = ev_in[idx_neg]
    out[idx_neg] = np.stack((np.ones_like(val_neg)*255, 255-val_neg*255, 255-val_neg*255), axis=1)
    return out

def merge_rgb_and_ev_img(rgb_img: np.ndarray, ev_img: np.ndarray):
    return np.clip(0.8*rgb_img + 0.2*ev_img, 0, 255).astype('uint8')

def compute_flow_error_map(flow_gt: torch.Tensor, flow_pred: torch.Tensor, valid_mask: Optional[torch.Tensor]=None):
    flow_gt = flow_torch2numpy(flow_gt)
    flow_pred = flow_torch2numpy(flow_pred)
    assert flow_gt.ndim == 3
    assert flow_gt.shape[-1] == 2
    assert flow_gt.shape == flow_pred.shape
    assert flow_gt.shape[:-1] == flow_pred.shape[:-1]
    diff = flow_pred - flow_gt
    diff = diff**2
    error = np.sqrt(np.sum(diff, axis=-1))
    if valid_mask is not None:
        assert is_cpu(valid_mask)
        valid_mask = valid_mask.numpy()
        assert valid_mask.ndim == 2
        assert flow_gt.shape[:-1] == valid_mask.shape
        error[~valid_mask] = 0
    return error, valid_mask

def compute_flow_error_img(flow_gt: torch.Tensor, flow_pred: torch.Tensor, valid_mask: Optional[torch.Tensor]=None, max_error: float=3.0):
    error_map, valid_mask = compute_flow_error_map(flow_gt, flow_pred, valid_mask)
    error_map[error_map > max_error] = max_error
    # reference: https://stackoverflow.com/questions/53235638/how-should-i-convert-a-float32-image-to-an-uint8-image/53236206
    cm = plt.get_cmap('coolwarm')
    error_img = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cm_img = cm(error_img)
    cm_img = cv2.normalize(cm_img[..., :3], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cm_img is RGB at this point which is fine for WandB

    if valid_mask is not None:
        invalid_indices = np.where(valid_mask == False)
        cm_img[invalid_indices[0], invalid_indices[1], :] = 0

    return cm_img

def create_summary_img(
        flow_pred: torch.Tensor,
        flow_gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor]=None,
        ev_repr_reduced: Optional[torch.Tensor]=None,
        ev_repr_reduced_m1: Optional[torch.Tensor]=None,
        images: Optional[List[torch.Tensor]]=None,
        max_error: float=3.0):
    summary_img_list = list()
    if images is not None:
        # Assuming here that we are only dealing with RAFT* + optional images
        assert len(images) == 2
        images = [batch_to_column(img).numpy() for img in images]
        images = [np.moveaxis(img, 0, -1) for img in images]

    if ev_repr_reduced_m1 is not None:
        ev_repr_reduced_m1_col = batch_to_column(ev_repr_reduced_m1)
        ev_img_m1 = ev_repr_reduced_to_img(ev_repr_reduced_m1_col)
        if images is not None:
            images_0 = images[0]
            ev_img_m1 = merge_rgb_and_ev_img(images_0, ev_img_m1)
        summary_img_list.append(ev_img_m1)
    else:
        assert images is not None
        summary_img_list.append(images[0])

    if ev_repr_reduced is not None:
        ev_repr_reduced_col = batch_to_column(ev_repr_reduced)
        ev_img = ev_repr_reduced_to_img(ev_repr_reduced_col)
        if images is not None:
            images_1 = images[1]
            ev_img = merge_rgb_and_ev_img(images_1, ev_img)
        summary_img_list.append(ev_img)
    else:
        assert images is not None
        summary_img_list.append(images[1])

    flow_pred_col = batch_to_column(flow_pred)
    flow_gt_col = batch_to_column(flow_gt)
    valid_col = batch_to_column(valid_mask) if valid_mask is not None else None

    flow_pred_img = flow_to_img(flow_pred_col)
    flow_gt_img = flow_to_img(flow_gt_col)
    flow_error_img = compute_flow_error_img(flow_pred_col, flow_gt_col, valid_col, max_error=max_error)

    summary_img_list.extend([flow_pred_img, flow_gt_img, flow_error_img])

    summary_img = np.hstack(summary_img_list)
    return summary_img


def get_grad_flow_figure(named_params):
    """Creates figure to visualize gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Use this function after loss.backwards()
    """
    data_dict = {
        'name': list(),
        'grad_abs': list(),
    }
    for name, param in named_params:
        if param.requires_grad and param.grad is not None:
            grad_abs = param.grad.abs()
            data_dict['name'].append(name)
            data_dict['grad_abs'].append(grad_abs.mean().cpu().item())

    data_frame = pd.DataFrame.from_dict(data_dict)

    fig = px.bar(data_frame, x='name', y='grad_abs')
    return fig


def add_bezier_to_figure(
        figure: go.Figure,
        bezier_curves: BezierCurves,
        bezier_batch_idx: int,
        background_img: np.ndarray,
        multi_flow_gt: Optional[List[torch.Tensor]]=None, # [(N, 2, H, W), ...] (M times)
        hide_axis: bool=True,
        row_idx: Optional[int]=None,
        col_idx: Optional[int]=None,
        num_x: int=15,
        num_y: int=10,
        num_t: int=10,
        y_flow_from_top: int=20,
        y_flow_from_bottom: int=50,
        x_flow_from_left: int=10,
        x_flow_from_right: int=10,
        x_add_margin: int=100,
        y_add_margin: int=100) -> None:

    # NOTE:
    # If we have multi_flow_gt we use that one as background because it's nice!
    if multi_flow_gt is not None and len(multi_flow_gt) > 1:
        flow_background = multi_flow_gt[-1][bezier_batch_idx]
        background_img = flow_to_img(flow_background)

    figure.add_trace(go.Image(
        z=background_img,
        opacity=0.45),
        row=row_idx,
        col=col_idx,
    )

    wd = bezier_curves.width
    ht = bezier_curves.height

    # coords0: 2, ht, wd -> coords0[0] gives x coordinates and coords0[1] y coordinates
    coords0 = np.stack(np.meshgrid(np.arange(0, wd), np.arange(0, ht)))

    have_gt = (multi_flow_gt is not None)
    coords1_list_gt = list()

    coords1_list_bez = list()
    time_values = list()
    for time in np.linspace(0, 1, num=num_t+1):
        time_values.append(time)
        flow = bezier_curves.get_flow_from_reference(time)[bezier_batch_idx].numpy()
        coords1_list_bez.append(coords0 + flow)

    show_linear = False

    if have_gt:
        coords1_list_gt.append(coords0)
        for multi_flow_gt_step in multi_flow_gt:
            coords1_list_gt.append(coords0 + multi_flow_gt_step[bezier_batch_idx].numpy())

    y_start = y_flow_from_top
    y_end = ht - 1 - y_flow_from_bottom
    assert y_start < y_end
    x_start = x_flow_from_left
    x_end = wd - 1 - x_flow_from_right
    assert x_start < x_end
    # TODO: potential to parallelize/speed up with plotly express?
    # TODO: parallelize coordinate reading with numpy?
    for x in np.linspace(x_start, x_end, num=num_x):
        x = int(x)
        for y in np.linspace(y_start, y_end, num=num_y):
            y = int(y)
            x_list = list()
            y_list = list()
            for coords1 in coords1_list_bez:
                x_list.append(coords1[0, y, x])
                y_list.append(coords1[1, y, x])
            figure.add_trace(go.Scatter(
                x=x_list,
                y=y_list,
                mode='lines+markers',
                # see: https://plotly.com/python-api-reference/generated/plotly.graph_objects.scatter.html#plotly.graph_objects.scatter.Line
                line=dict(color='blue', width=2),
                marker=dict(size=4, symbol='circle')),
                row=row_idx,
                col=col_idx,
            )
            # To show linear prediction
            if show_linear:
                figure.add_trace(go.Scatter(
                    x=[x_list[0], x_list[-1]],
                    y=[y_list[0], y_list[-1]],
                    mode='lines',
                    line=dict(color='limegreen', width=1.5, dash='solid')),
                    row=row_idx,
                    col=col_idx,
                )
            if have_gt:
                x_list_gt = list()
                y_list_gt = list()
                for coords1 in coords1_list_gt:
                    x_list_gt.append(coords1[0, y, x])
                    y_list_gt.append(coords1[1, y, x])
                figure.add_trace(go.Scatter(
                    x=x_list_gt,
                    y=y_list_gt,
                    mode='lines+markers',
                    # see: https://plotly.com/python-api-reference/generated/plotly.graph_objects.scatter.html#plotly.graph_objects.scatter.Line
                    line=dict(color='red', width=2),
                    marker=dict(size=4, symbol='circle')),
                    row=row_idx,
                    col=col_idx,
                )
            # To show starting point
            figure.add_trace(go.Scatter(
                x=[x_list[0]],
                y=[y_list[0]],
                mode='markers',
                # see: https://plotly.com/python-api-reference/generated/plotly.graph_objects.scatter.html#plotly.graph_objects.scatter.Line
                line=dict(color='black', width=2),
                marker=dict(size=4, symbol='circle')),
                row=row_idx,
                col=col_idx,
            )

    hide_axes_opts = dict()
    if hide_axis:
        hide_axes_opts.update({'visible': False})
    figure.update_xaxes(showgrid=False, range=(0 - x_add_margin, wd + x_add_margin), row=row_idx, col=col_idx, **hide_axes_opts)
    figure.update_yaxes(showgrid=False, range=(ht + y_add_margin, 0 - y_add_margin), scaleanchor='x', row=row_idx, col=col_idx, **hide_axes_opts)


def single_plot_bezier(
        bezier_curves: BezierCurves,
        bezier_batch_idx: int,
        background_img: np.ndarray,
        multi_flow_gt: Optional[List[torch.Tensor]]=None, # [(N, 2, H, W), ...] (M times)
        hide_axis=True,
        num_x: int=15,
        num_y: int=10,
        num_t: int=10,
        y_flow_from_top: int=20,
        y_flow_from_bottom: int=50,
        x_flow_from_left: int=10,
        x_flow_from_right: int=10,
        x_add_margin: int=100,
        y_add_margin: int=100) -> go.Figure:

    assert background_img.ndim ==3
    assert background_img.shape[-1] == 3

    figure = go.Figure()
    add_bezier_to_figure(
        figure,
        bezier_curves,
        bezier_batch_idx,
        background_img,
        multi_flow_gt=multi_flow_gt,
        hide_axis=hide_axis,
        row_idx=None,
        col_idx=None,
        num_x=num_x,
        num_y=num_y,
        num_t=num_t,
        y_flow_from_top=y_flow_from_top,
        y_flow_from_bottom=y_flow_from_bottom,
        x_flow_from_left=x_flow_from_left,
        x_flow_from_right=x_flow_from_right,
        x_add_margin=x_add_margin,
        y_add_margin=y_add_margin)
    figure.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='White',
        plot_bgcolor='White',
    )

    return figure


def multi_plot_bezier(
        bezier_curves: BezierCurves,
        background_imgs_nhwc: np.ndarray, # N, H, W, C
        hide_axis=True,
        num_cols: int=3,
        num_x: int=15,
        num_y: int=10,
        num_t: int=10,
        y_flow_from_top: int=20,
        y_flow_from_bottom: int=50,
        x_flow_from_left: int=10,
        x_flow_from_right: int=10,
        x_add_margin: int=100,
        y_add_margin: int=100) -> go.Figure:

    batch_size = bezier_curves.batch_size
    assert batch_size == background_imgs_nhwc.shape[0]
    assert background_imgs_nhwc.ndim == 4 # N, H, W, 3
    assert background_imgs_nhwc.shape[-1] == 3

    num_rows = math.ceil(batch_size/num_cols)

    figure = make_subplots(rows=num_rows, cols=num_cols)

    for batch_idx in range(batch_size):

        background_img = background_imgs_nhwc[batch_idx]

        row_idx = batch_idx//num_cols
        col_idx = batch_idx - row_idx*num_cols

        # plotly uses 1-based indexing
        row_idx += 1
        col_idx += 1
        add_bezier_to_figure(
            figure,
            bezier_curves,
            batch_idx,
            background_img,
            hide_axis=hide_axis,
            row_idx=row_idx,
            col_idx=col_idx,
            num_x=num_x,
            num_y=num_y,
            num_t=num_t,
            y_flow_from_top=y_flow_from_top,
            y_flow_from_bottom=y_flow_from_bottom,
            x_flow_from_left=x_flow_from_left,
            x_flow_from_right=x_flow_from_right,
            x_add_margin=x_add_margin,
            y_add_margin=y_add_margin)

    figure.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='White',
        plot_bgcolor='White',
    )
    return figure


def multi_plot_bezier_array(
        bezier_curves: BezierCurves,
        background_imgs_nhwc: np.ndarray, # N, H, W, C
        multi_flow_gt: Optional[List[torch.Tensor]]=None, # [(N, 2, H, W), ...] (M times)
        hide_axis: bool=True,
        num_cols: int=3,
        num_x: int=15,
        num_y: int=10,
        num_t: int=10,
        y_flow_from_top: int=10,
        y_flow_from_bottom: int=10,
        x_flow_from_left: int=10,
        x_flow_from_right: int=10,
        x_add_margin: int=100,
        y_add_margin: int=100) -> np.ndarray:

    batch_size = bezier_curves.batch_size
    assert batch_size == background_imgs_nhwc.shape[0]
    assert background_imgs_nhwc.ndim == 4 # N, H, W, 3
    assert background_imgs_nhwc.shape[-1] == 3

    tensor_list: List[torch.Tensor] = list()
    for batch_idx in range(batch_size):
        assert batch_idx < bezier_curves.batch_size

        background_img = background_imgs_nhwc[batch_idx]

        figure = single_plot_bezier(
            bezier_curves,
            batch_idx,
            background_img,
            multi_flow_gt=multi_flow_gt,
            hide_axis=hide_axis,
            num_x=num_x,
            num_y=num_y,
            num_t=num_t,
            y_flow_from_top=y_flow_from_top,
            y_flow_from_bottom=y_flow_from_bottom,
            x_flow_from_left=x_flow_from_left,
            x_flow_from_right=x_flow_from_right,
            x_add_margin=x_add_margin,
            y_add_margin=y_add_margin)

        array = plotly_fig2array(figure, scale=2)
        array = np.moveaxis(array, -1, 0) # h,w,c -> c,h,w
        tensor_list.append(torch.from_numpy(array))

    grid_tensor = make_grid(tensor_list, nrow=num_cols, padding=0, normalize=False)
    grid_array = grid_tensor.numpy()
    grid_array = np.moveaxis(grid_array, 0, -1) # c,h,w -> h,w,c
    return grid_array


def plotly_fig2array(
        fig: go.Figure,
        width: Optional[int]=None,
        height: Optional[int]=None,
        scale: Union[int, float]=1) -> np.ndarray:
    # docs: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.to_image
    fig_bytes = fig.to_image(format="png", width=width, height=height, scale=scale, engine='kaleido')
    buf = io.BytesIO(fig_bytes)
    with Image.open(buf, 'r') as img:
        img_array = np.asarray(img)
    return img_array
