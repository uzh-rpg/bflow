from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BezierHead(nn.Module):
    def __init__(self, bezier_degree: int, input_dim=128, hidden_dim=256):
        super().__init__()
        output_dim = bezier_degree * 2
        # TODO: figure out if we need to increase the capacity of this head due to bezier curves
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, model_params: Dict[str, Any], output_dim: int=128):
        super().__init__()
        cor_planes = self._num_cor_planes(model_params['correlation'], model_params['use_boundary_images'], model_params['use_events'])

        # TODO: Are two layers enough for this? Because the number of input channels grew substantially
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        c2_out = 192
        self.convc2 = nn.Conv2d(256, c2_out, 3, padding=1)

        # TODO: Consider upgrading the number of channels of the flow encoders (dealing now with bezier curves)
        bezier_planes = model_params['bezier_degree'] * 2
        self.convf1 = nn.Conv2d(bezier_planes, 128, 7, padding=3)
        f2_out = 64
        self.convf2 = nn.Conv2d(128, f2_out, 3, padding=1)

        combined_channels = f2_out+c2_out
        self.conv = nn.Conv2d(combined_channels, output_dim-bezier_planes, 3, padding=1)

    @staticmethod
    def _num_cor_planes(corr_params: Dict[str, Any], use_boundary_images: bool, use_events: bool):
        assert use_events or use_boundary_images
        out = 0
        if use_events:
            ev_params = corr_params['ev']
            ev_corr_levels = ev_params['levels']
            ev_corr_radius = ev_params['radius']
            assert len(ev_corr_levels) > 0
            assert len(ev_corr_radius) > 0
            assert len(ev_corr_levels) == len(ev_corr_radius)
            for lvl, rad in zip(ev_corr_levels, ev_corr_radius):
                out += lvl * (2*rad + 1)**2
        if use_boundary_images:
            img_corr_levels = corr_params['img']['levels']
            img_corr_radius = corr_params['img']['radius']
            out += img_corr_levels * (2*img_corr_radius + 1)**2
        return out

    def forward(self, bezier, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        bez = F.relu(self.convf1(bezier))
        bez = F.relu(self.convf2(bez))

        cor_bez = torch.cat([cor, bez], dim=1)

        out = F.relu(self.conv(cor_bez))
        return torch.cat([out, bezier], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, model_params: Dict[str, Any], hidden_dim: int=128):
        super().__init__()
        motion_encoder_output_dim = model_params['motion']['dim']
        context_dim = model_params['context']['dim']
        bezier_degree = model_params['bezier_degree']
        self.encoder = BasicMotionEncoder(model_params, output_dim=motion_encoder_output_dim)
        gru_input_dim = context_dim + motion_encoder_output_dim
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=gru_input_dim)
        self.bezier_head = BezierHead(bezier_degree, input_dim=hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, bezier):
        # TODO: check if we can simplify this similar to the DROID-SLAM update block
        motion_features = self.encoder(bezier, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_bezier = self.bezier_head(net)

        # scale mask to balance gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_bezier
