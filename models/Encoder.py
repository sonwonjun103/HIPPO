import torch
import torch.nn as nn
import torch.nn.functional as F

def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out

def Activation_layer(activation_cfg, inplace=True):
 
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out

class Conv3d_wd(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = (1,1,1),
                 padding = (0,0,0),
                 dilation = (1,1,1),
                 groups=1,
                 bias=False):
        super().__init__()
        
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1,1,1,1,1)
        weight = weight / std.expand_as(weight)
        
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3x3(in_planes,
              out_planes,
              kernel_size,
              stride=(1,1,1),
              padding=(0,0,0),
              dilation = (1,1,1),
              bias=False,
              weight_std=False):
    
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 inplanes,
                 planes,
                 norm_cfg,
                 activation_cfg,
                 stride=(1,1,1),
                 downsample=None,
                 weight_std=False):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False,
                            weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)
        return out

class CNNEncoder(nn.Module):
    arch_settings = {
        9 : (ResBlock, (2,2,1,1))
    }

    def __init__(self,
                 depth,
                 in_channels=1,
                 init_channels=64,
                 norm_cfg='BN',
                 activation_cfg='ReLU',
                 weight_std=False):
        super().__init__()

        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = init_channels
        self.conv1 = conv3x3x3(in_channels,
                                init_channels,
                                kernel_size=7,
                                stride=(2,2,2),
                                padding=3,
                                bias=False,
                                weight_std=weight_std)
        
        self.norm1 = Norm_layer(norm_cfg, init_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        
        self.channels = [128,256,512,1024]
        self.layer1 = self._make_layer(block, self.channels[0], layers[0], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, self.channels[1], layers[1], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, self.channels[2], layers[2], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer4 = self._make_layer(block, self.channels[3], layers[3], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []

        self.__initialize_weight()

    def __initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=(1,1,1),
                    norm_cfg='BN',
                    activation_cfg='ReLU',
                    weight_std=False):
        downsample = None
        if stride!=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3x3(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    weight_std=weight_std),

                Norm_layer(norm_cfg, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))

        return nn.Sequential(*layers)       
 
    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        out.append(x)

        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)

        return out    
    
if __name__=='__main__':
    sample = torch.randn(4, 1, 64, 128,128)
    model = CNNEncoder(depth=9)

    print(f"Model Parameter : {sum(p.numel() for p in model.parameters())}")
    
    pred = model(sample)
    print(f"1 layer {pred[0].shape}") # 4 64,32,64,64
    print(f"2 layer {pred[1].shape}") # 4 128,16,32,32
    print(f"3 layer {pred[2].shape}") # 4 256,8,16,16 
    print(f"4 layer {pred[3].shape}") # 4 512,4,8,8