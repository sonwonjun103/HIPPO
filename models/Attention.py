# spatial attention
# channel attention
# boundary attention

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

# ============================================================================== #

class Block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_features, in_features//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_features//2),
            nn.Conv3d(in_features//2, out_features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_features)
        )

        self.__initialize_weight()
 
    def __initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.conv(x)


class EAttention(nn.Module):
    def __init__(self,
                 in_features=64):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv = Block(in_features)

    def forward(self, x1, x2, x3, x4, edge=None):
        # x4
        x4 = self.upsample(x4)
        # x3
        x3 = self.upsample(x3 + x4)
        # x2
        x2 = self.upsample(x2 + x3)

        x2_ = self.upsample(x2)
        x1 = self.upsample(x1)
        x = x1 +  x2_
        
        if edge is None:
            edge = torch.ones(x.size())

        output = x * edge

        output = self.conv(output)
        
        b, c, d, h, w = output.size()
        output = output.view(b,c, -1)
        output = self.sigmoid(output)

        output = torch.reshape(output, (b, c, d, h, w))

        return output

# def mask_to_onehot(mask, num_classes=1):
#     _mask = [mask == i for i in range(0, num_classes+1)]
#     _mask = [np.expand_dims(x, 0) for x in _mask]
#     return np.concatenate(_mask, 0)

# def onehot_to_binary_edges(mask, radius=1, num_classes=2):
#     if radius < 0:
#         return mask

#     # We need to pad the borders for boundary conditions
#     mask_pad = np.pad(mask, ((0, 0), (0, 0), (0, 0),(1, 1), (1, 1), (1,1)), mode='constant', constant_values=0)

#     edgemap = np.zeros(mask.shape[1:])

#     for i in range(num_classes):
#         dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
#         dist = dist[:,:, 1:-1, 1:-1, 1:-1]
#         dist[dist > radius] = 0
#         edgemap += dist
#     edgemap = np.expand_dims(edgemap, axis=0)
#     edgemap = (edgemap > 0).astype(np.uint8)
#     return edgemap

# def mask_to_edges(mask):
#     _edge = mask
#     _edge = mask_to_onehot(_edge)
#     _edge = onehot_to_binary_edges(_edge)
#     return _edge

# class BAttention(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         reduction = x.shape[-1] // 16
        
#         feature_ = mask_to_edges(x)
#         feature_ = feature_[0][:,:,:, reduction:-reduction, reduction:-reduction, reduction:-reduction]
#         feature_ = np.pad(feature_, ( (0,0), (0,0), (reduction:-reduction), (reduction:-reduction), (reduction:-reduction) ) )
#         return feature_
    
# ============================================================================== #

class SAttention(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio,
                 pool_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.BatchNorm3d(gate_channels //  reduction_ratio),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool3d(pool_size)
        self.maxpool = nn.AdaptiveMaxPool3d(pool_size)

    def forward(self, x):
        x_avg_pool = self.avgpool(x)
        x_max_pool = self.maxpool(x)

        attention = torch.cat((x_avg_pool, x_max_pool), dim=1)
        attention = self.conv(attention)

        return x * attention

# ============================================================================== #

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class CAttention(nn.Module):
    def __init__(self,
                 gate_channels,
                 pool_size,
                 reduction_ratio=16):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels //  reduction_ratio, gate_channels),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool3d(pool_size)
        self.maxpool = nn.AdaptiveMaxPool3d(pool_size)

    def forward(self, x):
        # max, avg pooling 거치고
        # mlp 거치고
        # concat 해서
        # sigmoid 통과
        a = self.avgpool(x)
        m = self.maxpool(x)
        #print(f"a : {a.shape} m : {m.shape}")

        avg_pool = self.mlp(a)
        max_pool = self.mlp(m)

        #print(f"avg_pool : {avg_pool.shape} max_pool : {max_pool.shape}")

        attention = avg_pool + max_pool
        #attention = attention.view(4,-1, 1,2,2)
        #print(f"attention : {attention.shape}")
        attention = attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        #print(f"attention : {attention.shape}")

        return x * attention
    
if __name__=='__main__':
    sample = torch.randn(4, 64, 32, 64, 32)
    model = EAttention(64)
    print(f"Model Parameter : {sum(p.numel() for p in model.parameters())}")
    pred = model(sample)
    print(f"pred : {pred.shape}")