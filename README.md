# Dense Continuous-Time Optical Flow from Event Cameras

![readme](https://github.com/uzh-rpg/bflow/assets/6841681/2b8c7a7f-3c75-49d4-85cd-51c78b0884d3)

This is the official Pytorch implementation of the TPAMI 2024 paper [Dense Continuous-Time Optical Flow from Event Cameras](https://ieeexplore.ieee.org/document/10419040).

If you find this code useful, please cite us:
```bibtex
@Article{Gehrig2024pami,
  author        = {Mathias Gehrig and Manasi Muglikar and Davide Scaramuzza},
  title         = {Dense Continuous-Time Optical Flow from Event Cameras},
  journal       = {{IEEE} Trans. Pattern Anal. Mach. Intell. (T-PAMI)},
  year          = 2024
}
```

## Conda Installation
We highly recommend to use [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to reduce the installation time.
```Bash
conda create -y -n bflow python=3.11 pip
conda activate bflow
conda config --set channel_priority flexible

CUDA_VERSION=12.1

conda install -y h5py=3.10.0 blosc-hdf5-plugin=1.0.0 llvm-openmp=15.0.7 \
hydra-core=1.3.2 einops=0.7 tqdm numba \
pytorch=2.1.2 torchvision pytorch-cuda=$CUDA_VERSION \
-c pytorch -c nvidia -c conda-forge

python -m pip install pytorch-lightning==2.1.3 wandb==0.16.1 \
opencv-python==4.8.1.78 imageio==2.33.1 lpips==0.1.4 \
pandas==2.1.4 plotly==5.18.0 moviepy==1.0.3 tabulate==0.9.0 \
loguru==0.7.2 matplotlib==3.8.2 scikit-image==0.22.0 kaleido
```
## Data
### MultiFlow
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">Train</th>
<th valign="bottom">Val</th>
<tr><td align="left">pre-processed dataset</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/bflow/multiflow/train.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/bflow/multiflow/val.tar">download</a></td>
</tr>
</tbody></table>

### DSEC
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">Train</th>
<th valign="bottom">Test (input)</th>
<tr><td align="left">pre-processed dataset</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/bflow/DSEC/train.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/bflow/DSEC/test.tar">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>c1b618fc</tt></td>
<td align="center"><tt>ffbacb7e</tt></td>
</tr>
</tbody></table>

## Checkpoints

### MultiFlow

<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">Events only</th>
<th valign="bottom">Events + Images</th>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/bflow/checkpoints/multiflow/E_LU5_BD10.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/bflow/checkpoints/multiflow/E_I_LU5_BD10.ckpt">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>61e102</tt></td>
<td align="center"><tt>2ce3aa</tt></td>
</tr>
</tbody></table>

### DSEC

<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">Events only</th>
<th valign="bottom">Events + Images</th>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/bflow/checkpoints/dsec/E_LU4_BD2.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/bflow/checkpoints/dsec/E_I_LU4_BD2.ckpt">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>d17002</tt></td>
<td align="center"><tt>05770b</tt></td>
</tr>
</tbody></table>


## Training

### MultiFlow
- Set `DATA_DIR` as the path to the MultiFlow dataset (parent of train and val dir)
- Set
    - `MDL_CFG=E_I_LU5_BD10_lowpyramid` to use both events and frames, or
    - `MDL_CFG=E_LU5_BD10_lowpyramid` to use only events
- Set `LOG_ONLY_NUMBERS=true` to avoid logging images (can require a lot of space). Set to false by default.

```Bash
GPU_ID=0
python train.py model=raft-spline dataset=multiflow_regen dataset.path=${DATA_DIR} wandb.group_name=multiflow \
hardware.gpus=${GPU_ID} hardware.num_workers=6 +experiment/multiflow/raft_spline=${MLD_CFG} \
logging.only_numbers=${LOG_ONLY_NUMBERS}
```

### DSEC
- Set `DATA_DIR` as the path to the DSEC dataset (parent of train and test dir)
- 
- Set
  - `MDL_CFG=E_I_LU4_BD2_lowpyramid` to use both events and frames, or
  - `MDL_CFG=E_LU4_BD2_lowpyramid` to use only events
- Set `LOG_ONLY_NUMBERS=true` to avoid logging images (can require a lot of space). Set to false by default.

```Bash
GPU_ID=0
python train.py model=raft-spline dataset=dsec dataset.path=${DATA_DIR} wandb.group_name=dsec \
hardware.gpus=${GPU_ID} hardware.num_workers=6 +experiment/dsec/raft_spline=${MLD_CFG} \
logging.only_numbers=${LOG_ONLY_NUMBERS}
```

## Evaluation 

### MultiFlow
- Set `DATA_DIR` as the path to the MultiFlow dataset (parent of train and val dir)
- Set
  - `MDL_CFG=E_I_LU5_BD10_lowpyramid` to use both events and frames, or
  - `MDL_CFG=E_LU5_BD10_lowpyramid` to use only events
- Set `CKPT` to the path of the correct checkpoint

```Bash
GPU_ID=0
python val.py model=raft-spline dataset=multiflow_regen dataset.path=${DATA_DIR} hardware.gpus=${GPU_ID} \
+experiment/multiflow/raft_spline=${MLD_CFG} checkpoint=${CKPT}
```

### DSEC

work in progress

## Code Acknowledgments
This project has used code from [RAFT](https://github.com/princeton-vl/RAFT) for parts of the model architecture.
