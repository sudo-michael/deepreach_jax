# DeepReach Jax
Implementation of [Deepreach](https://github.com/smlbansal/deepreach) in Jax / Flax

## Installation
```
conda env create -f environment.yml
conda activate deepreach
```

## Example
```
python experiment_scripts/hji_air3d --wandb --experiment-name air3d-jax
```

## Jax vs PyTorch
There's a slight speed up compared to the PyTorch implementaiton.
|  |Jax   |PyTorch | 
|---|---|---|
|Air3D| 48m   |  79m |

Tested on:

CPU: AMD Ryzen 9 3900X

GPU: NVIDIA 3080