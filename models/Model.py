import torch
import torch.nn as nn

from models.Encoder import CNNEncoder
from models.Attention import SAttention, CAttention, EAttention
from models.MRFB import MRFB

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

class ConCat(nn.Module):
    def __init__(self,
                 in_features):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv1 = Block(in_features)
        self.conv2 = Block(in_features)
        self.conv3 = Block(in_features)

        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x2, x3, x4, edge):
        x2 = self.conv1(x2)
        x3 = self.conv2(x3)
        x4 = self.conv3(x4)

        x4 = self.upsample(x4)
        x3 = self.upsample(x3 + x4)
        x2 = self.upsample(x2 + x3)
        x2 = self.upsample(x2)

        output = x2 + edge

        # b, c, d, h, w = output.size()
        # output = output.view(b,c , -1)
        # output = self.softmax(output)
        # output = torch.reshape(output, (b, c, d, h, w))

        return output

class Model(nn.Module):
    def __init__(self,
                 in_features,
                 mrfb_features,
                 reduction_ratio=16,
                 pool_size=(1,1,1),
                 depth=9):
        super().__init__()

        self.encoder = CNNEncoder(depth=depth)
        self.MRFB1 = MRFB(in_features[0], 
                         mrfb_features)
        self.MRFB2 = MRFB(in_features[1], 
                         mrfb_features)
        self.MRFB3 = MRFB(in_features[2], 
                         mrfb_features)

        # self.sattention = SAttention(in_features,
        #                              pool_size,
        #                              reduction_ratio)
        
        self.cattention1 = CAttention(mrfb_features,
                                     pool_size,
                                     reduction_ratio)
        self.cattention2 = CAttention(mrfb_features,
                                     pool_size,
                                     reduction_ratio)
        self.cattention3 = CAttention(mrfb_features,
                                     pool_size,
                                     reduction_ratio)
        
        self.eattention = EAttention()
        self.conv = ConCat(mrfb_features)
        
    def forward(self, x, edge=None): # B, 1, 96, 128, 128
        x1, x2, x3, x4 = self.encoder(x)

        x2_ = self.MRFB1(x2)
        x3_ = self.MRFB2(x3)
        x4_ = self.MRFB3(x4)
        
        # Edge attention
        output_edge = self.eattention(x1, x2_, x3_, x4_, edge)

        # channel attention 
        x2_c = self.cattention1(x2_)
        x3_c = self.cattention2(x3_)
        x4_c = self.cattention3(x4_)

        output = self.conv(x2_c, x3_c, x4_c, output_edge)

        return output, output_edge

if __name__=='__main__':
    sample = torch.randn(4, 1, 96, 128, 128)
    edge = torch.randn(4, 1, 96, 128, 128)

    model = Model([128,256,512],
                  64)
    print(f"Model Parameter : {sum(p.numel() for p in model.parameters())}")
    pred = model(sample, edge)