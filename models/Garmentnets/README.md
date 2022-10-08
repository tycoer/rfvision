# GarmentNets
This model is a mm-style modification of GarmentNets.

For more details, please refer to:
* Code : https://github.com/columbia-ai-robotics/garmentnets
* Paper : https://arxiv.org/abs/2104.05177
* Website: https://garmentnets.cs.columbia.edu/

## Preparations
Garmentnets needs to install additional libs:
* [zarr](https://github.com/zarr-developers/zarr-python/tree/main)
* [igl](https://github.com/libigl/libigl-python-bindings)
* [torch_scatter](https://github.com/rusty1s/pytorch_scatter)


## Dataset
Download datasets from this [page](https://github.com/columbia-ai-robotics/garmentnets)

## Train
Garmentnets need a two-stage training, 
1. Train the configuration file [garmentnet_nocs_cfg.py](./configs/garmentnet_nocs_cfg.py)
2. Train the configuration file [garmentnet_wnf_cfg.py](./configs/garmentnet_wnf_cfg.py)


Due to missing implementation of validation "--no-validate" should be used in training command.\
Refer to [README.md]((../../README.md))

## Test
Refer to [README.md](../../README.md)