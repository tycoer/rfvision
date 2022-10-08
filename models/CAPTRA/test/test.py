from mmdet3d.models.backbones import PointNet2SASSG
import torch
if __name__ == '__main__':
    t = torch.rand(2, 4096, 3).cuda(1)
    m = PointNet2SASSG(in_channels=3).cuda(1)
    res = m(t)