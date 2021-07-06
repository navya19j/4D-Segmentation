import numpy
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchio as tio


class Double_Conv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Double_Conv,self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )

    def forward(self,x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, features = [64,128,256,512]):
        #super calls initialization of inheritance class
        super(UNet,self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        #Down part of the UNET #encoder
        for i in range (0,len(features)):
            self.downs.append(Double_Conv(in_channels,features[i]))
            in_channels = features[i]

        #Up part of the UNET
        for i in range (len(features)-1,-1,-1):
            feature = features[i]
            self.ups.append(nn.ConvTranspose3d(feature*2,feature,kernel_size=2,stride=2))
            self.ups.append(Double_Conv(feature*2,feature))
        
        self.last_layer = Double_Conv(features[-1],features[-1]*2)

        self.final_conv = nn.Conv3d(features[0], out_channels,kernel_size=1)

    def forward(self,x):
        
        #channels = 3D
        # = (torch.empty(batch_size,depth,channels,height,width)).float()
        skip_connection = []
        for down in self.downs:
            x = down(x)
            skip_connection.append(x)
            x = self.max_pool(x)
        x = self.last_layer(x)
            
            #check = numpy.array(skip_connection)
            #print(check.shape)
        skip_connection = skip_connection[::-1]

        for i in range (0,len(self.ups)):
            x = self.ups[i](x)
            #print(x.shape)
            if (i%2 == 0):
                #print(True)
                skip_new = skip_connection[int(i/2)]
                if x.shape != skip_new.shape:
                    x = TF.interpolate(x, size = skip_new.shape[2:])
                concat = torch.cat((skip_new,x),dim=1)
                x = concat
            #print(x.shape)
        return self.final_conv(x)