import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))


class UpBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(UpBlock,self).__init__()
        self.add_module('up', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))


class DownBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(DownBlock,self).__init__()
        self.add_module('down', nn.MaxPool2d(2))
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd_old(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        print("here!!!")
        self.once = True
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self,x,y):
        if self.once == True:
            print(f'GENERATOR input = {x.shape}')
        x = self.head(x)
        if self.once == True:
            print(f'GENERATOR head output = {x.shape}')
        x = self.body(x)
        if self.once == True:
            print(f'GENERATOR body output = {x.shape}')
        x = self.tail(x)
        if self.once == True:
            print(f'GENERATOR tail output = {x.shape}')
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]

        if self.once == True:
            self.once = False
        return x+y



class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        # self.body = nn.Sequential()
        # for i in range(opt.num_layer-2):
        #     N = int(opt.nfc/pow(2,(i+1)))
        #     block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
        #     self.body.add_module('block%d'%(i+1),block)

        self.body.add_module('down1', DownBlock(max(2*int(opt.nfc/pow(2,1)), opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1))
        # self.body.add_module('down2', DownBlock(max(2*int(opt.nfc/pow(2,2)), opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1))
        self.body.add_module('up1', UpBlock(max(2*int(opt.nfc/pow(2,2)), opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1))
        # self.body.add_module('up2', UpBlock(max(2*int(opt.nfc/pow(2,4)), opt.min_nfc),max(N,opt.min_nfc),(4, 5),1,1))

        self.body1 = DownBlock(max(2*int(opt.nfc/pow(2,1)), opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
        self.body2 = UpBlock(max(2*int(opt.nfc/pow(2,2)), opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            # nn.Upsample(size = , mode='bilinear', align_corners=True)
            nn.Tanh()
        )

        self.once = True
        self.final_size = None
        self.N = N
        self.opt = opt

    def forward(self,x,y):

        if self.once:
            print(f'GENERATOR input = {x.shape}')
            self.final_size = (x.shape[2] - 10, x.shape[3] - 10)
            print('final size', self.final_size)
            self.tail = nn.Sequential(
                nn.Conv2d(max(self.N,self.opt.min_nfc),self.opt.nc_im,kernel_size=self.opt.ker_size,stride =1,padding=self.opt.padd_size),
                nn.Upsample(size = (self.final_size), mode='bilinear', align_corners=True),
                nn.Tanh()
            ).to('cuda:0')
        x = self.head(x)
        if self.once:
            print(f'GENERATOR head output = {x.shape}')
        # x = self.body(x)
        
        embedding = self.body1(x)
        x = self.body2(embedding)

        if self.once:
            print(f'GENERATOR body output = {x.shape}')
        x = self.tail(x)
        if self.once:
            print(f'GENERATOR tail output = {x.shape}')
            print(f'y shape = {y.shape}')

        # if (y.shape[2]-x.shape[2]) % 2 == 0 and (y.shape[3]-x.shape[3]) % 2 == 0:
        #     ind1 = int((y.shape[2]-x.shape[2])/2)
        #     ind2 = int((y.shape[3]-x.shape[3])/2)

        #     y = y[:,:,ind1:(y.shape[2]-ind1),ind2:(y.shape[3]-ind2)]
        
        # elif (y.shape[2]-x.shape[2]) % 2 == 0 and (y.shape[3]-x.shape[3]) % 2 != 0:
        #     ind1 = int((y.shape[2]-x.shape[2])/2)
        #     ind2 = int((y.shape[3]-x.shape[3])/2)

        #     y = y[:,:,ind1:(y.shape[2]-ind1),ind2:(y.shape[3]-ind2-1)]
        
        # elif (y.shape[2]-x.shape[2]) % 2 != 0 and (y.shape[3]-x.shape[3]) % 2 == 0:
        #     ind1 = int((y.shape[2]-x.shape[2])/2)
        #     ind2 = int((y.shape[3]-x.shape[3])/2)

        #     y = y[:,:,ind1:(y.shape[2]-ind1-1),ind2:(y.shape[3]-ind2)]
        
        # elif (y.shape[2]-x.shape[2]) % 2 != 0 and (y.shape[3]-x.shape[3]) % 2 != 0:
        #     ind1 = int((y.shape[2]-x.shape[2])/2)
        #     ind2 = int((y.shape[3]-x.shape[3])/2)

        
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]

        if self.once:
            self.once = False
        return x+y, embedding
