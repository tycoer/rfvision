import torch
import os
import argparse

rf_nocs_coordnet_keys = ['coordnet.backbone.sa1.mlps.0.layer0.conv.weight',
 'coordnet.backbone.sa1.mlps.0.layer0.conv.bias',
 'coordnet.backbone.sa1.mlps.0.layer0.bn.weight',
 'coordnet.backbone.sa1.mlps.0.layer0.bn.bias',
 'coordnet.backbone.sa1.mlps.0.layer0.bn.running_mean',
 'coordnet.backbone.sa1.mlps.0.layer0.bn.running_var',
 'coordnet.backbone.sa1.mlps.0.layer0.bn.num_batches_tracked',
 'coordnet.backbone.sa1.mlps.0.layer1.conv.weight',
 'coordnet.backbone.sa1.mlps.0.layer1.conv.bias',
 'coordnet.backbone.sa1.mlps.0.layer1.bn.weight',
 'coordnet.backbone.sa1.mlps.0.layer1.bn.bias',
 'coordnet.backbone.sa1.mlps.0.layer1.bn.running_mean',
 'coordnet.backbone.sa1.mlps.0.layer1.bn.running_var',
 'coordnet.backbone.sa1.mlps.0.layer1.bn.num_batches_tracked',
 'coordnet.backbone.sa1.mlps.0.layer2.conv.weight',
 'coordnet.backbone.sa1.mlps.0.layer2.conv.bias',
 'coordnet.backbone.sa1.mlps.0.layer2.bn.weight',
 'coordnet.backbone.sa1.mlps.0.layer2.bn.bias',
 'coordnet.backbone.sa1.mlps.0.layer2.bn.running_mean',
 'coordnet.backbone.sa1.mlps.0.layer2.bn.running_var',
 'coordnet.backbone.sa1.mlps.0.layer2.bn.num_batches_tracked',
 'coordnet.backbone.sa1.mlps.1.layer0.conv.weight',
 'coordnet.backbone.sa1.mlps.1.layer0.conv.bias',
 'coordnet.backbone.sa1.mlps.1.layer0.bn.weight',
 'coordnet.backbone.sa1.mlps.1.layer0.bn.bias',
 'coordnet.backbone.sa1.mlps.1.layer0.bn.running_mean',
 'coordnet.backbone.sa1.mlps.1.layer0.bn.running_var',
 'coordnet.backbone.sa1.mlps.1.layer0.bn.num_batches_tracked',
 'coordnet.backbone.sa1.mlps.1.layer1.conv.weight',
 'coordnet.backbone.sa1.mlps.1.layer1.conv.bias',
 'coordnet.backbone.sa1.mlps.1.layer1.bn.weight',
 'coordnet.backbone.sa1.mlps.1.layer1.bn.bias',
 'coordnet.backbone.sa1.mlps.1.layer1.bn.running_mean',
 'coordnet.backbone.sa1.mlps.1.layer1.bn.running_var',
 'coordnet.backbone.sa1.mlps.1.layer1.bn.num_batches_tracked',
 'coordnet.backbone.sa1.mlps.1.layer2.conv.weight',
 'coordnet.backbone.sa1.mlps.1.layer2.conv.bias',
 'coordnet.backbone.sa1.mlps.1.layer2.bn.weight',
 'coordnet.backbone.sa1.mlps.1.layer2.bn.bias',
 'coordnet.backbone.sa1.mlps.1.layer2.bn.running_mean',
 'coordnet.backbone.sa1.mlps.1.layer2.bn.running_var',
 'coordnet.backbone.sa1.mlps.1.layer2.bn.num_batches_tracked',
 'coordnet.backbone.sa1.mlps.2.layer0.conv.weight',
 'coordnet.backbone.sa1.mlps.2.layer0.conv.bias',
 'coordnet.backbone.sa1.mlps.2.layer0.bn.weight',
 'coordnet.backbone.sa1.mlps.2.layer0.bn.bias',
 'coordnet.backbone.sa1.mlps.2.layer0.bn.running_mean',
 'coordnet.backbone.sa1.mlps.2.layer0.bn.running_var',
 'coordnet.backbone.sa1.mlps.2.layer0.bn.num_batches_tracked',
 'coordnet.backbone.sa1.mlps.2.layer1.conv.weight',
 'coordnet.backbone.sa1.mlps.2.layer1.conv.bias',
 'coordnet.backbone.sa1.mlps.2.layer1.bn.weight',
 'coordnet.backbone.sa1.mlps.2.layer1.bn.bias',
 'coordnet.backbone.sa1.mlps.2.layer1.bn.running_mean',
 'coordnet.backbone.sa1.mlps.2.layer1.bn.running_var',
 'coordnet.backbone.sa1.mlps.2.layer1.bn.num_batches_tracked',
 'coordnet.backbone.sa1.mlps.2.layer2.conv.weight',
 'coordnet.backbone.sa1.mlps.2.layer2.conv.bias',
 'coordnet.backbone.sa1.mlps.2.layer2.bn.weight',
 'coordnet.backbone.sa1.mlps.2.layer2.bn.bias',
 'coordnet.backbone.sa1.mlps.2.layer2.bn.running_mean',
 'coordnet.backbone.sa1.mlps.2.layer2.bn.running_var',
 'coordnet.backbone.sa1.mlps.2.layer2.bn.num_batches_tracked',
 'coordnet.backbone.sa2.mlps.0.layer0.conv.weight',
 'coordnet.backbone.sa2.mlps.0.layer0.conv.bias',
 'coordnet.backbone.sa2.mlps.0.layer0.bn.weight',
 'coordnet.backbone.sa2.mlps.0.layer0.bn.bias',
 'coordnet.backbone.sa2.mlps.0.layer0.bn.running_mean',
 'coordnet.backbone.sa2.mlps.0.layer0.bn.running_var',
 'coordnet.backbone.sa2.mlps.0.layer0.bn.num_batches_tracked',
 'coordnet.backbone.sa2.mlps.0.layer1.conv.weight',
 'coordnet.backbone.sa2.mlps.0.layer1.conv.bias',
 'coordnet.backbone.sa2.mlps.0.layer1.bn.weight',
 'coordnet.backbone.sa2.mlps.0.layer1.bn.bias',
 'coordnet.backbone.sa2.mlps.0.layer1.bn.running_mean',
 'coordnet.backbone.sa2.mlps.0.layer1.bn.running_var',
 'coordnet.backbone.sa2.mlps.0.layer1.bn.num_batches_tracked',
 'coordnet.backbone.sa2.mlps.0.layer2.conv.weight',
 'coordnet.backbone.sa2.mlps.0.layer2.conv.bias',
 'coordnet.backbone.sa2.mlps.0.layer2.bn.weight',
 'coordnet.backbone.sa2.mlps.0.layer2.bn.bias',
 'coordnet.backbone.sa2.mlps.0.layer2.bn.running_mean',
 'coordnet.backbone.sa2.mlps.0.layer2.bn.running_var',
 'coordnet.backbone.sa2.mlps.0.layer2.bn.num_batches_tracked',
 'coordnet.backbone.sa2.mlps.1.layer0.conv.weight',
 'coordnet.backbone.sa2.mlps.1.layer0.conv.bias',
 'coordnet.backbone.sa2.mlps.1.layer0.bn.weight',
 'coordnet.backbone.sa2.mlps.1.layer0.bn.bias',
 'coordnet.backbone.sa2.mlps.1.layer0.bn.running_mean',
 'coordnet.backbone.sa2.mlps.1.layer0.bn.running_var',
 'coordnet.backbone.sa2.mlps.1.layer0.bn.num_batches_tracked',
 'coordnet.backbone.sa2.mlps.1.layer1.conv.weight',
 'coordnet.backbone.sa2.mlps.1.layer1.conv.bias',
 'coordnet.backbone.sa2.mlps.1.layer1.bn.weight',
 'coordnet.backbone.sa2.mlps.1.layer1.bn.bias',
 'coordnet.backbone.sa2.mlps.1.layer1.bn.running_mean',
 'coordnet.backbone.sa2.mlps.1.layer1.bn.running_var',
 'coordnet.backbone.sa2.mlps.1.layer1.bn.num_batches_tracked',
 'coordnet.backbone.sa2.mlps.1.layer2.conv.weight',
 'coordnet.backbone.sa2.mlps.1.layer2.conv.bias',
 'coordnet.backbone.sa2.mlps.1.layer2.bn.weight',
 'coordnet.backbone.sa2.mlps.1.layer2.bn.bias',
 'coordnet.backbone.sa2.mlps.1.layer2.bn.running_mean',
 'coordnet.backbone.sa2.mlps.1.layer2.bn.running_var',
 'coordnet.backbone.sa2.mlps.1.layer2.bn.num_batches_tracked',
 'coordnet.backbone.sa3.mlps.0.layer0.conv.weight',
 'coordnet.backbone.sa3.mlps.0.layer0.conv.bias',
 'coordnet.backbone.sa3.mlps.0.layer0.bn.weight',
 'coordnet.backbone.sa3.mlps.0.layer0.bn.bias',
 'coordnet.backbone.sa3.mlps.0.layer0.bn.running_mean',
 'coordnet.backbone.sa3.mlps.0.layer0.bn.running_var',
 'coordnet.backbone.sa3.mlps.0.layer0.bn.num_batches_tracked',
 'coordnet.backbone.sa3.mlps.0.layer1.conv.weight',
 'coordnet.backbone.sa3.mlps.0.layer1.conv.bias',
 'coordnet.backbone.sa3.mlps.0.layer1.bn.weight',
 'coordnet.backbone.sa3.mlps.0.layer1.bn.bias',
 'coordnet.backbone.sa3.mlps.0.layer1.bn.running_mean',
 'coordnet.backbone.sa3.mlps.0.layer1.bn.running_var',
 'coordnet.backbone.sa3.mlps.0.layer1.bn.num_batches_tracked',
 'coordnet.backbone.sa3.mlps.0.layer2.conv.weight',
 'coordnet.backbone.sa3.mlps.0.layer2.conv.bias',
 'coordnet.backbone.sa3.mlps.0.layer2.bn.weight',
 'coordnet.backbone.sa3.mlps.0.layer2.bn.bias',
 'coordnet.backbone.sa3.mlps.0.layer2.bn.running_mean',
 'coordnet.backbone.sa3.mlps.0.layer2.bn.running_var',
 'coordnet.backbone.sa3.mlps.0.layer2.bn.num_batches_tracked',
 'coordnet.backbone.fp3.mlps.layer0.conv.weight',
 # 'coordnet.backbone.fp3.mlps.layer0.conv.bias',
 'coordnet.backbone.fp3.mlps.layer0.bn.weight',
 'coordnet.backbone.fp3.mlps.layer0.bn.bias',
 'coordnet.backbone.fp3.mlps.layer0.bn.running_mean',
 'coordnet.backbone.fp3.mlps.layer0.bn.running_var',
 'coordnet.backbone.fp3.mlps.layer0.bn.num_batches_tracked',
 'coordnet.backbone.fp3.mlps.layer1.conv.weight',
 # 'coordnet.backbone.fp3.mlps.layer1.conv.bias',
 'coordnet.backbone.fp3.mlps.layer1.bn.weight',
 'coordnet.backbone.fp3.mlps.layer1.bn.bias',
 'coordnet.backbone.fp3.mlps.layer1.bn.running_mean',
 'coordnet.backbone.fp3.mlps.layer1.bn.running_var',
 'coordnet.backbone.fp3.mlps.layer1.bn.num_batches_tracked',
 'coordnet.backbone.fp2.mlps.layer0.conv.weight',
 # 'coordnet.backbone.fp2.mlps.layer0.conv.bias',
 'coordnet.backbone.fp2.mlps.layer0.bn.weight',
 'coordnet.backbone.fp2.mlps.layer0.bn.bias',
 'coordnet.backbone.fp2.mlps.layer0.bn.running_mean',
 'coordnet.backbone.fp2.mlps.layer0.bn.running_var',
 'coordnet.backbone.fp2.mlps.layer0.bn.num_batches_tracked',
 'coordnet.backbone.fp2.mlps.layer1.conv.weight',
 # 'coordnet.backbone.fp2.mlps.layer1.conv.bias',
 'coordnet.backbone.fp2.mlps.layer1.bn.weight',
 'coordnet.backbone.fp2.mlps.layer1.bn.bias',
 'coordnet.backbone.fp2.mlps.layer1.bn.running_mean',
 'coordnet.backbone.fp2.mlps.layer1.bn.running_var',
 'coordnet.backbone.fp2.mlps.layer1.bn.num_batches_tracked',
 'coordnet.backbone.fp1.mlps.layer0.conv.weight',
 # 'coordnet.backbone.fp1.mlps.layer0.conv.bias',
 'coordnet.backbone.fp1.mlps.layer0.bn.weight',
 'coordnet.backbone.fp1.mlps.layer0.bn.bias',
 'coordnet.backbone.fp1.mlps.layer0.bn.running_mean',
 'coordnet.backbone.fp1.mlps.layer0.bn.running_var',
 'coordnet.backbone.fp1.mlps.layer0.bn.num_batches_tracked',
 'coordnet.backbone.fp1.mlps.layer1.conv.weight',
 # 'coordnet.backbone.fp1.mlps.layer1.conv.bias',
 'coordnet.backbone.fp1.mlps.layer1.bn.weight',
 'coordnet.backbone.fp1.mlps.layer1.bn.bias',
 'coordnet.backbone.fp1.mlps.layer1.bn.running_mean',
 'coordnet.backbone.fp1.mlps.layer1.bn.running_var',
 'coordnet.backbone.fp1.mlps.layer1.bn.num_batches_tracked',
 'coordnet.backbone.conv1.weight',
 'coordnet.backbone.conv1.bias',
 'coordnet.backbone.bn1.weight',
 'coordnet.backbone.bn1.bias',
 'coordnet.backbone.bn1.running_mean',
 'coordnet.backbone.bn1.running_var',
 'coordnet.backbone.bn1.num_batches_tracked',
 'coordnet.seg_head.0.weight',
 'coordnet.seg_head.0.bias',
 'coordnet.nocs_head.0.weight',
 'coordnet.nocs_head.0.bias',
 'coordnet.nocs_head.1.weight',
 'coordnet.nocs_head.1.bias',
 'coordnet.nocs_head.1.running_mean',
 'coordnet.nocs_head.1.running_var',
 'coordnet.nocs_head.1.num_batches_tracked',
 'coordnet.nocs_head.3.weight',
 'coordnet.nocs_head.3.bias']

rf_nocs_rotnet_keys = ['rotnet.regress_net.encoder.sa1.mlps.0.layer0.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer0.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer0.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer0.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer0.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer1.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer1.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer1.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer1.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer1.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer2.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer2.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer2.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer2.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer2.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer2.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.0.layer2.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer0.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer0.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer0.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer0.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer0.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer1.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer1.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer1.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer1.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer1.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer2.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer2.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer2.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer2.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer2.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer2.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.1.layer2.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer0.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer0.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer0.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer0.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer0.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer1.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer1.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer1.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer1.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer1.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer2.conv.weight',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer2.conv.bias',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer2.bn.weight',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer2.bn.bias',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer2.bn.running_mean',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer2.bn.running_var',
 'rotnet.regress_net.encoder.sa1.mlps.2.layer2.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer0.conv.weight',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer0.conv.bias',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer0.bn.weight',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer0.bn.bias',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer0.bn.running_var',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer1.conv.weight',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer1.conv.bias',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer1.bn.weight',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer1.bn.bias',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer1.bn.running_var',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer2.conv.weight',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer2.conv.bias',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer2.bn.weight',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer2.bn.bias',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer2.bn.running_mean',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer2.bn.running_var',
 'rotnet.regress_net.encoder.sa2.mlps.0.layer2.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer0.conv.weight',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer0.conv.bias',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer0.bn.weight',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer0.bn.bias',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer0.bn.running_var',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer1.conv.weight',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer1.conv.bias',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer1.bn.weight',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer1.bn.bias',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer1.bn.running_var',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer2.conv.weight',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer2.conv.bias',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer2.bn.weight',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer2.bn.bias',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer2.bn.running_mean',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer2.bn.running_var',
 'rotnet.regress_net.encoder.sa2.mlps.1.layer2.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer0.conv.weight',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer0.conv.bias',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer0.bn.weight',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer0.bn.bias',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer0.bn.running_var',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer1.conv.weight',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer1.conv.bias',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer1.bn.weight',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer1.bn.bias',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer1.bn.running_var',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer2.conv.weight',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer2.conv.bias',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer2.bn.weight',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer2.bn.bias',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer2.bn.running_mean',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer2.bn.running_var',
 'rotnet.regress_net.encoder.sa3.mlps.0.layer2.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.fp3.mlps.layer0.conv.weight',
 # 'rotnet.regress_net.encoder.fp3.mlps.layer0.conv.bias',
 'rotnet.regress_net.encoder.fp3.mlps.layer0.bn.weight',
 'rotnet.regress_net.encoder.fp3.mlps.layer0.bn.bias',
 'rotnet.regress_net.encoder.fp3.mlps.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.fp3.mlps.layer0.bn.running_var',
 'rotnet.regress_net.encoder.fp3.mlps.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.fp3.mlps.layer1.conv.weight',
 # 'rotnet.regress_net.encoder.fp3.mlps.layer1.conv.bias',
 'rotnet.regress_net.encoder.fp3.mlps.layer1.bn.weight',
 'rotnet.regress_net.encoder.fp3.mlps.layer1.bn.bias',
 'rotnet.regress_net.encoder.fp3.mlps.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.fp3.mlps.layer1.bn.running_var',
 'rotnet.regress_net.encoder.fp3.mlps.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.fp2.mlps.layer0.conv.weight',
 # 'rotnet.regress_net.encoder.fp2.mlps.layer0.conv.bias',
 'rotnet.regress_net.encoder.fp2.mlps.layer0.bn.weight',
 'rotnet.regress_net.encoder.fp2.mlps.layer0.bn.bias',
 'rotnet.regress_net.encoder.fp2.mlps.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.fp2.mlps.layer0.bn.running_var',
 'rotnet.regress_net.encoder.fp2.mlps.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.fp2.mlps.layer1.conv.weight',
 # 'rotnet.regress_net.encoder.fp2.mlps.layer1.conv.bias',
 'rotnet.regress_net.encoder.fp2.mlps.layer1.bn.weight',
 'rotnet.regress_net.encoder.fp2.mlps.layer1.bn.bias',
 'rotnet.regress_net.encoder.fp2.mlps.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.fp2.mlps.layer1.bn.running_var',
 'rotnet.regress_net.encoder.fp2.mlps.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.fp1.mlps.layer0.conv.weight',
 # 'rotnet.regress_net.encoder.fp1.mlps.layer0.conv.bias',
 'rotnet.regress_net.encoder.fp1.mlps.layer0.bn.weight',
 'rotnet.regress_net.encoder.fp1.mlps.layer0.bn.bias',
 'rotnet.regress_net.encoder.fp1.mlps.layer0.bn.running_mean',
 'rotnet.regress_net.encoder.fp1.mlps.layer0.bn.running_var',
 'rotnet.regress_net.encoder.fp1.mlps.layer0.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.fp1.mlps.layer1.conv.weight',
 # 'rotnet.regress_net.encoder.fp1.mlps.layer1.conv.bias',
 'rotnet.regress_net.encoder.fp1.mlps.layer1.bn.weight',
 'rotnet.regress_net.encoder.fp1.mlps.layer1.bn.bias',
 'rotnet.regress_net.encoder.fp1.mlps.layer1.bn.running_mean',
 'rotnet.regress_net.encoder.fp1.mlps.layer1.bn.running_var',
 'rotnet.regress_net.encoder.fp1.mlps.layer1.bn.num_batches_tracked',
 'rotnet.regress_net.encoder.conv1.weight',
 'rotnet.regress_net.encoder.conv1.bias',
 'rotnet.regress_net.encoder.bn1.weight',
 'rotnet.regress_net.encoder.bn1.bias',
 'rotnet.regress_net.encoder.bn1.running_mean',
 'rotnet.regress_net.encoder.bn1.running_var',
 'rotnet.regress_net.encoder.bn1.num_batches_tracked',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.0.weight',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.0.bias',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.1.weight',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.1.bias',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.3.weight',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.3.bias',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.4.weight',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.4.bias',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.6.weight',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.6.bias',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.7.weight',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.7.bias',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.9.weight',
 'rotnet.regress_net.pose_pred.rtvec_head.0.model.9.bias']

def count_weights(state_dict):
    total_para = 0
    for i in state_dict.values():
        total_para += i.numel()
    return total_para

def main(offical_ckpt_path, out_path):
    offical_ckpt = torch.load(offical_ckpt_path)

    weights = list(offical_ckpt['model'].values())
    keys = list(offical_ckpt['model'].keys())
    if len(keys) == 189: # nocs rotnet
        rf_keys = rf_nocs_rotnet_keys
    elif len(keys) == 198: # nocs coordnet 197
        rf_keys = rf_nocs_coordnet_keys

    conv_interval = 2
    bn_interval = 5
    conv_start = 0
    bn_start = 18

    rf_weights = []
    while True:
        # sa1
        rf_weights.extend(weights[conv_start: conv_start + conv_interval])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval
        if bn_start >= 62:
            conv_start = 63
            bn_start = 75
            break

    while True:
        # sa2
        rf_weights.extend(weights[conv_start: conv_start + conv_interval])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval

        if bn_start >= 104:
            conv_start = 105
            bn_start = 111
            break

    while True:
        # sa3
        rf_weights.extend(weights[conv_start: conv_start + conv_interval])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval

        if bn_start >= 125:
            break

    if len(keys) == 189: # rotnet
        conv_start = 126
        bn_start = 130
    else: # coordnet
        # for i in range(126, 138):  # gen
        #     rf_weights.append(weights[i])
        conv_start = 138
        bn_start = 142


    while True:
        # fp3
        weights[conv_start] = weights[conv_start].unsqueeze(-1)
        rf_weights.extend([weights[conv_start]])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval
        if len(keys) == 189:
            if bn_start >= 139:
                conv_start = 140
                bn_start = 144
                break

        else:
            if bn_start >= 151:
                conv_start = 152
                bn_start = 156
                break

    while True:
        # fp2
        weights[conv_start] = weights[conv_start].unsqueeze(-1)
        rf_weights.extend([weights[conv_start]])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval
        if len(keys) == 189:
            if bn_start >= 153:
                conv_start = 154
                bn_start = 158
                break
        else:
            if bn_start >= 165:
                conv_start = 166
                bn_start = 170
                break

    while True:
        # fp1
        weights[conv_start] = weights[conv_start].unsqueeze(-1)
        rf_weights.extend([weights[conv_start]])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval
        if len(keys) == 189:
            if bn_start >= 167:
                break
        else:
            if bn_start >= 179:
                break

    if len(keys) == 189: # rotnet
        for i in range(168, len(weights)):
            rf_weights.append(weights[i])
    else: # coordnet
        for i in range(180, len(weights)):
            rf_weights.append(weights[i])

    rf_ckpt = dict(state_dict = {k: v for k, v in zip(rf_keys, rf_weights)},
                meta={'epoch': 250},
                optimizer={})
    torch.save(rf_ckpt, out_path)
    print(f'rfvision-style checkpoint is saved to {out_path}')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Convert offical checkpoint file to rfvision-style.')
    arg_parser.add_argument('--checkpoint', required=True, type=str, help='offical checkpoint file path.',
                            default='/home/hanyang/CAPTRA-main/6_mug_rot/ckpt/model_0000.pt')
    arg_parser.add_argument('--out_path', type=str, help='out path of rfvision-style checkpoint.', required=True)
    args = arg_parser.parse_args()

    main(args.checkpoint, args.out_path)