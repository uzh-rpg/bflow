import torch
import torch.nn.functional as F


def bilinear_sampler(img, coords):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # img (corr): batch*h1*w1, 1, h2, w2
    # coords: batch*h1*w1, 2*r+1, 2*r+1, 2
    H, W = img.shape[-2:]
    # *grid: batch*h1*w1, 2*r+1, 2*r+1, 1
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # map *grid from [0, N-1] to [-1, 1]
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    # grid: batch*h1*w1, 2*r+1, 2*r+1, 2
    grid = torch.cat([xgrid, ygrid], dim=-1)
    # img: batch*h1*w1, 1, 2*r+1, 2*r+1
    img = F.grid_sample(img, grid, align_corners=True)

    return img


def coords_grid(batch, ht, wd, device):
    # ((ht, wd), (ht, wd))
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    # 2, ht, wd
    coords = torch.stack(coords[::-1], dim=0).float()
    # batch, 2, ht, wd
    return coords[None].repeat(batch, 1, 1, 1)


def cvx_upsample(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ Upsample data [N, dim, H/8, W/8] -> [N, dim, H, W] using convex combination """
    N, dim, H, W = data.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    # NOTE: multiply by 8 due to the change in resolution.
    up_data = F.unfold(8 * data, [3, 3], padding=1)
    up_data = up_data.view(N, dim, 9, 1, 1, H, W)

    # N, dim, 8, 8, H, W
    up_data = torch.sum(mask * up_data, dim=2)
    # N, dim, H, 8, W, 8
    up_data = up_data.permute(0, 1, 4, 2, 5, 3)
    # N, dim, H*8, W*8
    return up_data.reshape(N, dim, 8*H, 8*W)
