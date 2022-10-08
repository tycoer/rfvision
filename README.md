# RFVision
## Train
### For single gpu training:
```
python tools/train.py ${config}
```

Example:
```
python tools/train.py models/garmentnet/configs/garmentnet_nocs_cfg.py
```


### For multiple gpus training:
```
bash tools/dist_train.sh ${config} ${num_gpus}
```
Example:

```
bash tools/dist_train.sh models/garmentnet/configs/garmentnet_nocs_cfg.py 4
```

## Test
### For single gpu testing
```
python tools/test.py ${config} ${checkpoint}
```
Example
```
python tools/test.py models/garmentnet/configs/garmentnet_nocs_cfg.py xxx.pth
```

### For multiple gpus testing:
```
bash tools/dist_test.sh ${config} ${checkpoint} ${num_gpus}
```
Example:

```
bash tools/dist_test.sh models/garmentnet/configs/garmentnet_nocs_cfg.py xxx.pth 4
```