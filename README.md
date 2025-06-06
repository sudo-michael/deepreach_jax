# DeepReach Jax
Implementation of [Deepreach](https://github.com/smlbansal/deepreach) in Jax / Flax

## Installation
```
uv sync
```

## Example
```
uv run hji_air3d.py --wandb --experiment-name air3d-jax
```

## Jax vs PyTorch
There's a slight speed up compared to the PyTorch implementaiton.
|  |Jax   |PyTorch | 
|---|---|---|
|Air3D| 48m   |  79m |

Tested on:

CPU: AMD Ryzen 9 3900X

GPU: NVIDIA 3080