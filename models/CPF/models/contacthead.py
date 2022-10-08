import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ContactHead(nn.Module):
    def __init__(self, out_dim, base_neurons=None):
        super().__init__()
        if base_neurons is None:
            base_neurons = [65, 512, 512, 512]
        assert len(base_neurons) >= 1

        # returns k for each object vert
        self.out_dim = out_dim

        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Conv2d(inp_neurons, out_neurons, kernel_size=(1, 1), stride=1, padding=0))
            layers.append(nn.ReLU())
        self.final_layer = nn.Conv2d(out_neurons, self.out_dim, kernel_size=(1, 1), stride=1, padding=0)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        return out


class VertexContactHead(nn.Module):
    def __init__(self, base_neurons=None, out_dim=1):
        super().__init__()
        if base_neurons is None:
            base_neurons = [65, 512, 512, 65]
        assert len(base_neurons) >= 1

        # returns k for each object vert
        self.out_dim = out_dim

        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Conv2d(inp_neurons, out_neurons, kernel_size=(1, 1), stride=1, padding=0))
            layers.append(nn.ReLU())
        self.final_layer = nn.Conv2d(out_neurons, self.out_dim, kernel_size=(1, 1), stride=1, padding=0)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        # // out = self.sigmoid(out)
        return out


class PointNetContactHead(nn.Module):
    def __init__(self, feat_dim=65, n_region=17, n_anchor=4):
        super().__init__()

        # record input feature dimension
        self.feat_dim = feat_dim

        # returns k for each object vert
        self.n_region = n_region
        self.n_anchor = n_anchor

        # encode module
        self.encoder = PointNetEncodeModule(self.feat_dim)
        self._concat_feat_dim = self.encoder.dim_out
        self.vertex_contact_decoder = PointNetDecodeModule(self._concat_feat_dim, 1)
        self.contact_region_decoder = PointNetDecodeModule(self._concat_feat_dim + 1, self.n_region)
        self.anchor_elasti_decoder = PointNetDecodeModule(self._concat_feat_dim + 17, self.n_anchor)

    def forward(self, inp):
        # inp = TENSOR[NBATCH, 65, NPOINT, 1]
        batch_size, _, n_point, _ = inp.shape
        feat = inp.squeeze(3)  # TENSOR[NBATCH, 65, NPOINT]
        concat_feat = self.encoder(feat)  # TENSOR[NBATCH, 4992, NPOINT]
        vertex_contact = self.vertex_contact_decoder(concat_feat)  # TENSOR[NBATCH, 1, NPOINT]
        contact_region = self.contact_region_decoder(concat_feat, vertex_contact)  # TENSOR[NBATCH, 17, NPOINT]
        anchor_elasti = self.anchor_elasti_decoder(concat_feat, contact_region)  # TENSOR[NBATCH, 4, NPOINT]
        # post process
        vertex_contact = vertex_contact.squeeze(1).contiguous()
        contact_region = contact_region.transpose(1, 2).contiguous()  # TENSOR[NBATCH, NPOINT, 17]
        anchor_elasti = anchor_elasti.transpose(1, 2).contiguous()  # TENSOR[NBATCH, NPOINT, 4]

        # !: here sigmoid is compulsory, since we use bce (instead of  bce_with_logit)
        anchor_elasti = torch.sigmoid(anchor_elasti)
        return vertex_contact, contact_region, anchor_elasti



# * currently, STN is unused
class STNkd(nn.Module):
    def __init__(self, k=65):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(k, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        device = x.device
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(batchsize, 1)
        )
        iden = iden.to(device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# * currently, regularizer on STN transforms is unused
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    device = trans.device
    I = torch.eye(d, device=device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


class PointNetEncodeModule(nn.Module):
    def __init__(self, dim_in=65):
        super().__init__()
        self.dim_in = dim_in
        # self.stn1 = STNkd(k=channel)
        self.conv1 = torch.nn.Conv1d(dim_in, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        # self.stn2 = STNkd(k=128)
        self.dim_out = 4992

    def forward(self, point_cloud):
        # pointcloud = TENSOR[NBATCH, 65, NPOINT]
        B, D, N = point_cloud.size()

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))  # TENSOR[NBATCH, 128, NPOINT]
        out2 = F.relu(self.bn2(self.conv2(out1)))  # TENSOR[NBATCH, 128, NPOINT]
        out3 = F.relu(self.bn3(self.conv3(out2)))  # TENSOR[NBATCH, 128, NPOINT]
        out4 = F.relu(self.bn4(self.conv4(out3)))  # TENSOR[NBATCH, 512, NPOINT]
        out5 = self.bn5(self.conv5(out4))  # TENSOR[NBATCH, 2048, NPOINT]
        out_max = torch.max(out5, 2, keepdim=True)[0]  # TENSOR[NBATCH, 2048, 1]
        expand = out_max.repeat(1, 1, N)  # TENSOR[NBATCH, 2048, NPOINT]
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)  # TENSOR[NBATCH, 4992, NPOINT]

        return concat


class PointNetDecodeModule(nn.Module):
    def __init__(self, dim_in=4992, dim_out=1):
        super().__init__()
        self.convs1 = torch.nn.Conv1d(dim_in, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, dim_out, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, pointnet_feat, extra_feat=None):
        # pointnet_feat = TENSOR[NBATCH, 4944, NPOINT]
        # extra_feat = TENSOR[NBATCH, 1, NPOINT]
        if extra_feat is not None:
            pointnet_feat = torch.cat((pointnet_feat, extra_feat), dim=1)
        net = F.relu(self.bns1(self.convs1(pointnet_feat)))  # TENSOR[NBATCH, 256, NPOINT]
        net = F.relu(self.bns2(self.convs2(net)))  # TENSOR[NBATCH, 256, NPOINT]
        net = F.relu(self.bns3(self.convs3(net)))  # TENSOR[NBATCH, 128, NPOINT]
        net = self.convs4(net)  # TENSOR[NBATCH, 1, NPOINT]
        return net
