import torch 
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.__initialize_weight()

    def __initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x

class MRFB(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = nn.Sequential(
            BasicBlock(in_channel, 
                       self.out_channel,
                       1)
        )

        self.branch1 = nn.Sequential(
            BasicBlock(self.in_channel, self.out_channel, 1),
            BasicBlock(self.out_channel, self.out_channel, kernel_size=(1,3,3), padding=(0,1,1)),
            BasicBlock(self.out_channel, self.out_channel, kernel_size=(3,3,1), padding=(1,1,0))
        )

        self.branch2 = nn.Sequential(
            BasicBlock(self.in_channel, self.out_channel, 1),
            BasicBlock(self.out_channel, self.out_channel, kernel_size=(1,5,5), padding=(0,2,2)),
            BasicBlock(self.out_channel, self.out_channel, kernel_size=(5,5,1), padding=(2,2,0)),
            BasicBlock(self.out_channel, self.out_channel, 3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicBlock(self.in_channel, self.out_channel, 1),
            BasicBlock(self.out_channel, self.out_channel, kernel_size=(1,7,7), padding=(0,3,3)),
            BasicBlock(self.out_channel, self.out_channel, kernel_size=(7,7,1), padding=(3,3,0)),
            BasicBlock(self.out_channel, self.out_channel, 3, padding=7, dilation=7)
        )       

        self.conv_cat = BasicBlock(4*self.out_channel, self.out_channel, 3, padding=1)
        self.conv_res = BasicBlock(in_channel, self.out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat([x0, x1, x2, x3], 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x
    
if __name__=='__main__':
    sample = torch.randn(4,256,4,8,8)
    model = MRFB(256)
    
    pred = model(sample)
    print(pred.shape)
    # print(f"1 layer {pred[0].shape}") # 4 64,32,64,64
    # print(f"2 layer {pred[1].shape}") # 4 128,16,32,32
    # print(f"3 layer {pred[2].shape}") # 4 256,8,16,16 
    # print(f"4 layer {pred[3].shape}") # 4 512,4,8,8