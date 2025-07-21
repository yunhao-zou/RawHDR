import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import arch_util

from torchvision import models
from arch_util import ResidualBlock_noBN
from uformer_block import Uformer

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 3, 3, 1, 1)
        self.vgg = models.vgg16(pretrained=False)
        self.device = device
        # for param in self.vgg.parameters():
            # param.requires_grad = False
        self.skip1 = LayerActivation(self.vgg.features, 3)
        self.skip2 = LayerActivation(self.vgg.features, 8)
        self.skip3 = LayerActivation(self.vgg.features, 15)
        self.skip4 = LayerActivation(self.vgg.features, 22)
        self.skip5 = LayerActivation(self.vgg.features, 29)

    def forward(self, x):
        x0 = x  # newly added
        x = self.conv1(x)
        self.vgg(x)

        return x, self.skip1.features.to(self.device), self.skip2.features.to(self.device) \
            , self.skip3.features.to(self.device), self.skip4.features.to(self.device), self.skip5.features.to(self.device),


def upsample(x, convT, skip, conv1x1, device):
    x = convT(x)
    bn = nn.BatchNorm2d(x.shape[1]).to(device)
    x = bn(x)
    x = F.leaky_relu(x, 0.2)

    skip = torch.log(skip ** 2 + 1.0/255.0)
    x = torch.cat([x, skip], dim=1)
    x = conv1x1(x)
    return x


def upsample_last(x, conv1x1_64_3, skip, conv1x1_6_3, device):
    x = conv1x1_64_3(x)
    bn = nn.BatchNorm2d(x.shape[1]).to(device)
    x = bn(x)
    x = F.leaky_relu(x, 0.2)

    skip = torch.log(skip ** 2 + 1.0 / 255.0)
    x = torch.cat([x, skip], dim=1)
    x = conv1x1_6_3(x)
    return x


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.latent_representation = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.convTranspose_5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_5 = Conv2d(1024, 512, kernel_size=1)

        self.convTranspose_4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_4 = Conv2d(1024, 512, kernel_size=1)

        self.convTranspose_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv1x1_3 = Conv2d(512, 256, kernel_size=1)

        self.convTranspose_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv1x1_2 = Conv2d(256, 128, kernel_size=1)

        self.convTranspose_1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv1x1_1 = Conv2d(128, 64, kernel_size=1)

        self.conv1x1_64_3 = Conv2d(64, 4, kernel_size=1)
        self.conv1x1_6_3 = Conv2d(7, 4, kernel_size=1)

    def forward(self, skip0, skip1, skip2, skip3, skip4, skip5):
        x = self.latent_representation(skip5)
        x = upsample(x, self.convTranspose_5, skip5, self.conv1x1_5, self.device)
        x = upsample(x, self.convTranspose_4, skip4, self.conv1x1_4, self.device)
        x = upsample(x, self.convTranspose_3, skip3, self.conv1x1_3, self.device)
        x = upsample(x, self.convTranspose_2, skip2, self.conv1x1_2, self.device)
        x = upsample(x, self.convTranspose_1, skip1, self.conv1x1_1, self.device)
        x = upsample_last(x, self.conv1x1_64_3, skip0, self.conv1x1_6_3, self.device)
        return x


class HDRCNN(nn.Module):
    def __init__(self, device):
        super(HDRCNN, self).__init__()
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)

    def forward(self, x):
        x = x.float()
        skip0, skip1, skip2, skip3, skip4, skip5 = self.encoder(x)
        x = self.decoder(skip0, skip1, skip2, skip3, skip4, skip5)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UNet, self).__init__()
        
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv10_1 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        conv10= self.conv10_1(conv9)
        # out = nn.functional.pixel_shuffle(conv10, 2)
        out = conv10
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt



class HDRUNet(nn.Module):

    def __init__(self, in_nc=4, out_nc=4, nf=64, act_type='relu'):
        super(HDRUNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.SFT_layer1 = arch_util.SFTLayer()
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        
        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 2)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 8)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 2)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.SFT_layer2 = arch_util.SFTLayer()
        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        cond_in_nc=4
        cond_nf=64
        self.cond_first = nn.Sequential(nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True), 
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), 
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True))
        self.CondNet1 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1))
        self.CondNet2 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1))
        self.CondNet3 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 3, 2, 1))

        self.mask_est = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1), 
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, nf, 3, 1, 1),
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, nf, 1),
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, out_nc, 1),
                                     )

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # x[0]: img; x[1]: cond
        mask = self.mask_est(x)

        cond = self.cond_first(x)   
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)

        fea0 = self.act(self.conv_first(x))

        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        out, _ = self.recon_trunk2((fea2, cond3))
        out = out + fea2

        out = self.act(self.up_conv1(out)) + fea1
        out, _ = self.recon_trunk3((out, cond2))

        out = self.act(self.up_conv2(out)) + fea0
        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = mask * x[0] + out
        return out


class cnnblock(nn.Module):
    def __init__(self,in_channle,out_channle):
        super(cnnblock,self).__init__()
        self.cnn_conv1=nn.Conv2d(in_channle,out_channle,3,1,1)
        self.ac1=nn.LeakyReLU(inplace = True)

        self.cnn_conv2=nn.Conv2d(out_channle,out_channle,3,1,1)
        self.ac2=nn.LeakyReLU(inplace = True)
    
    def forward(self,x):
        x=self.cnn_conv1(x)
        x=self.ac1(x)
        x=self.cnn_conv2(x)
        x=self.ac2(x)
        return x

class Upsample(nn.Module):
    """Upscaling"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels,out_channels,3,1,1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ac=nn.LeakyReLU(inplace = True)

    def forward(self, x, shape1, shape2):
        x = self.up(x)
        # input is CHW
        diffY = shape1 - x.shape[2]
        diffX = shape2 - x.shape[3]
        if self.bilinear:
            x = self.conv(x)
        x = self.ac(x)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        return x


class SubNet_3layers(nn.Module):
    def __init__(self, firstoutputchannl = 64):
        super(SubNet_3layers,self).__init__()
        self.outputchannl = 3
        self.maxpool=nn.MaxPool2d(2)
        self.block1=cnnblock(3,firstoutputchannl)
        self.block2=cnnblock(firstoutputchannl,2*firstoutputchannl)        
        self.block3=cnnblock(2*firstoutputchannl,4*firstoutputchannl)
        self.block4=cnnblock(4*firstoutputchannl,8*firstoutputchannl)
        self.up1=Upsample(8*firstoutputchannl,4*firstoutputchannl)
        self.block5=cnnblock(8*firstoutputchannl,4*firstoutputchannl)
        self.up2=Upsample(4*firstoutputchannl,2*firstoutputchannl)
        self.block6=cnnblock(4*firstoutputchannl,2*firstoutputchannl)
        self.up3=Upsample(2*firstoutputchannl,firstoutputchannl)
        self.block7=cnnblock(2*firstoutputchannl,firstoutputchannl)
        self.finalconv=nn.Conv2d(firstoutputchannl,self.outputchannl,1,1,0)

    def forward(self,x):
        out1=self.block1(x)
        out2=self.block2(self.maxpool(out1))
        out3=self.block3(self.maxpool(out2))
        out4=self.block4(self.maxpool(out3))
        in5=torch.cat([self.up1(out4,out3.shape[2],out3.shape[3]),out3],1)
        out5=self.block5(in5)
        in6=torch.cat([self.up2(out5,out2.shape[2],out2.shape[3]),out2],1)
        out6=self.block6(in6)
        in7=torch.cat([self.up3(out6,out1.shape[2],out1.shape[3]),out1],1)
        out7=self.block7(in7)
        predict= self.finalconv(out7)
        return predict

class SubNet_4layers(nn.Module):
    def __init__(self, firstoutputchannl = 64):
        super(SubNet_4layers,self).__init__()
        self.outputchannl = 3
        self.block1=cnnblock(3,firstoutputchannl)
        self.maxpool=nn.MaxPool2d(2)
        self.block2=cnnblock(firstoutputchannl,2*firstoutputchannl)        
        self.block3=cnnblock(2*firstoutputchannl,4*firstoutputchannl)
        self.block4=cnnblock(4*firstoutputchannl,8*firstoutputchannl)
        self.block5=cnnblock(8*firstoutputchannl,16*firstoutputchannl)

        self.up1=Upsample(16*firstoutputchannl,8*firstoutputchannl)
        self.block6=cnnblock(16*firstoutputchannl,8*firstoutputchannl)

        self.up2=Upsample(8*firstoutputchannl,4*firstoutputchannl)
        self.block7=cnnblock(8*firstoutputchannl,4*firstoutputchannl)

        self.up3=Upsample(4*firstoutputchannl,2*firstoutputchannl)
        self.block8=cnnblock(4*firstoutputchannl,2*firstoutputchannl)

        self.up4=Upsample(2*firstoutputchannl,firstoutputchannl)
        self.block9=cnnblock(2*firstoutputchannl,firstoutputchannl)
        self.finalconv=nn.Conv2d(firstoutputchannl,self.outputchannl,1,1,0)

    def forward(self,x):
        out1=self.block1(x)
        out2=self.block2(self.maxpool(out1))
        out3=self.block3(self.maxpool(out2))
        out4=self.block4(self.maxpool(out3))
        out5=self.block5(self.maxpool(out4))
        in6=torch.cat([self.up1(out5,out4.shape[2],out4.shape[3]),out4],1)
        out6=self.block6(in6)
        in7=torch.cat([self.up2(out6,out3.shape[2],out3.shape[3]),out3],1)
        out7=self.block7(in7)
        in8=torch.cat([self.up3(out7,out2.shape[2],out2.shape[3]),out2],1)
        out8=self.block8(in8)
        in9=torch.cat([self.up4(out8,out1.shape[2],out1.shape[3]),out1],1)
        out9=self.block9(in9)
        predict=self.finalconv(out9)
        return predict

class MSPEC_Net(nn.Module):
    def __init__ (self):
        super (MSPEC_Net,self).__init__()
        self.subnet1 = SubNet_4layers(24)
        self.subnet2 = SubNet_3layers(24)
        self.subnet3 = SubNet_3layers(24)
        self.subnet4 = SubNet_3layers(16)
        self.up1 = Upsample(3,3)
        self.up2 = Upsample(3,3)
        self.up3 = Upsample(3,3)
    def forward(self,L_list):
        y1_0 = self.subnet1(L_list[0])
        y1_1 = self.up1(y1_0,L_list[1].shape[2],L_list[1].shape[3])
        y2_0 = self.subnet2(y1_1 + L_list[1]) + y1_1
        y2_1 = self.up2(y2_0,L_list[2].shape[2],L_list[2].shape[3])
        y3_0 = self.subnet3(y2_1 + L_list[2]) + y2_1
        y3_1 = self.up3(y3_0,L_list[3].shape[2],L_list[3].shape[3])
        y4 = self.subnet4(y3_1 + L_list[3]) + y3_1
        Y_list = [y1_1,y2_1,y3_1,y4]
        return Y_list

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(4,8,4,2,1)
        self.ac1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(8,16,4,2,1)
        self.bn2 = nn.BatchNorm2d(16)
        self.ac2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(16,32,4,2,1)
        self.ac3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(32,64,4,2,1)
        self.ac4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(64,128,4,2,1)
        self.ac5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(128,128,4,2,1)
        self.ac6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(128,256,4,2,1)
        self.ac7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(256,1,2,2,0)
    def forward(self,x):
        if x.shape[2]!=256 and x.shape[3]!=256:
            x = F.interpolate(x,(256,256),mode='bilinear',align_corners=True)
        y = self.ac1(self.conv1(x))
        y = self.ac2(self.bn2(self.conv2(y)))
        y = self.ac3(self.conv3(y))
        y = self.ac4(self.conv4(y))
        y = self.ac5(self.conv5(y))
        y = self.ac6(self.conv6(y))
        y = self.ac7(self.conv7(y))
        y = self.conv8(y)
        return y

class DCENet(nn.Module):
    '''https://li-chongyi.github.io/Proj_Zero-DCE.html'''

    def __init__(self, n=8, return_results=[4, 6, 8]):
        '''
        Args
        --------
          n: number of iterations of LE(x) = LE(x) + alpha * LE(x) * (1-LE(x)).
          return_results: [4, 8] => return the 4-th and 8-th iteration results.
        '''
        super().__init__()
        self.n = n
        self.ret = return_results

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        self.conv7 = nn.Conv2d(64, 4 * n, kernel_size=3, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))

        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))

        out5 = self.relu(self.conv5(torch.cat((out4, out3), 1)))
        out6 = self.relu(self.conv6(torch.cat((out5, out2), 1)))

        alpha_stacked = self.tanh(self.conv7(torch.cat((out6, out1), 1)))

        alphas = torch.split(alpha_stacked, 4, 1)
        results = [x]
        for i in range(self.n):
            # x = x + alphas[i] * (x - x**2)  # as described in the paper
            # sign doesn't really matter becaus of symmetry.
            x = x + alphas[i] * (torch.pow(x, 2) - x)
            if i + 1 in self.ret:
                results.append(x)

        return results, alpha_stacked

class UnetEncoder(nn.Module):
    def __init__(self, input_channel=4):
        super(UnetEncoder, self).__init__()
        self.conv1_1 = nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        return conv1, conv2, conv3, pool3


class UnetDecoder(nn.Module):
    def __init__(self, output_channel=4):
        super(UnetDecoder, self).__init__()
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv5 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv5_1 = nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv6_1 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv7_1 = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(32, output_channel, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU()
    
    def forward(self, x1, x2):
        conv11, conv12, conv13, pool13 = x1
        conv21, conv22, conv23, pool23 = x2

        conv4 = self.lrelu(self.conv4_1(torch.cat([pool13, pool23], 1)))
        conv4 = self.lrelu(self.conv4_2(conv4))
        
        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv13, conv23], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv12, conv22], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv11, conv21], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        
        conv8 = self.conv8_1(conv7)

        return conv8

class RawHDR(nn.Module):
    def __init__(self, input_size=512, RB_gudie=True, G_guidance=True, RGB=False, softmask=True, softblending=False):
        super(RawHDR, self).__init__()
        self.RGB = RGB
        self.softmask = softmask
        self.softblending = softblending
        if self.RGB:
            in_channels = 3
            out_channels = 3
            g_channels = 1
        else:
            in_channels = 4
            out_channels = 4
            g_channels = 2
        self.mask = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), ResidualBlock_noBN(32), ResidualBlock_noBN(32), nn .Conv2d(32, 2, 3, 1, 1), nn.Sigmoid())
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.RB_guidance = RB_gudie
        self.G_guidance = G_guidance
        self.encoder_rgbg = UnetEncoder(in_channels)
        if self.RB_guidance:
            self.encoder_rb = UnetEncoder(2)
            self.decoder_rb = UnetDecoder(out_channels)
        if self.G_guidance:
            self.encoder_g = UnetEncoder(g_channels)
            self.decoder_g = UnetDecoder(out_channels)

        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.gsg = Uformer(img_size=input_size, in_chans=in_channels, dd_in=in_channels, embed_dim=16, depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), ResidualBlock_noBN(32), ResidualBlock_noBN(32), nn .Conv2d(32, out_channels, 3, 1, 1))
    
    def forward(self, x):
        if self.softmask:
            if self.softblending:
                max_channel = torch.max(x, dim=1)[0]
                min_channel = torch.min(x, dim=1)[0]
                mask_over = torch.max(torch.tensor(0.0), max_channel - 0.8) / 0.2
                mask_under = torch.max(torch.tensor(0.0), 0.2 - min_channel) / 0.2
            else:
                max_channel = torch.max(x, dim=1)[0]
                min_channel = torch.min(x, dim=1)[0]
                mask = self.mask(x)
                mask_over = mask[:, 0] #+ torch.max(torch.tensor(0.0), max_channel - 0.6) / 0.4
                mask_under = mask[:, 1] #+ torch.max(torch.tensor(0.0), 0.005 - min_channel) / 0.005
        else:
            max_channel = torch.max(x, dim=1)[0]
            min_channel = torch.min(x, dim=1)[0]
            mask_over = torch.zeros_like(max_channel)
            mask_over[max_channel>0.8] = 1
            mask_under = torch.zeros_like(min_channel)
            mask_under[min_channel<0.2] = 1
        I_RB = x[:, [0, 2], :, :]
        I_G = x[:, [1], :, :] if self.RGB else x[:, [1, 3], :, :]
        Y_RGBG = self.encoder_rgbg(x)
        if self.RB_guidance:
            Y_RB = self.encoder_rb(I_RB)
            Y_RB = self.decoder_rb(Y_RB, Y_RGBG)
        Y_RB = torch.einsum('bchw,bhw->bchw', Y_RB, mask_over) if self.RB_guidance else torch.zeros_like(x)
        if self.G_guidance:
            Y_G = self.encoder_g(I_G)
            Y_G = self.decoder_g(Y_G, Y_RGBG)
        Y_G = torch.einsum('bchw,bhw->bchw', Y_G, mask_under) if self.G_guidance else torch.zeros_like(x)
        Y_DI = Y_RB + Y_G
        Y_SG = self.gsg(x)
        out = self.final_conv(Y_DI + Y_SG)
        return out, mask_over, mask_under
        
class RawHDR_woTR(nn.Module):
    def __init__(self, input_size=512):
        super(RawHDR_woTR, self).__init__()
        self.mask = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), ResidualBlock_noBN(32), ResidualBlock_noBN(32), nn .Conv2d(32, 2, 3, 1, 1), nn.Sigmoid())
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder_rb = UnetEncoder(2)
        self.encoder_g = UnetEncoder(2)
        self.encoder_rgbg = UnetEncoder(4)
        self.decoder_rb = UnetDecoder(4)
        self.decoder_g = UnetDecoder(4)
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.final_conv = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), ResidualBlock_noBN(32), ResidualBlock_noBN(32), nn .Conv2d(32, 4, 3, 1, 1))
    
    def forward(self, x):
        mask = self.mask(x)
        mask_over = mask[:, 0]
        mask_under = mask[:, 1]
        I_RB = x[:, [0, 2], :, :]
        I_G = x[:, [1, 3], :, :]
        Y_RB = self.encoder_rb(I_RB)
        Y_G = self.encoder_g(I_G)
        Y_RGBG = self.encoder_rgbg(x)
        Y_RB = self.decoder_rb(Y_RB, Y_RGBG)
        Y_G = self.decoder_g(Y_G, Y_RGBG)
        Y_DI = torch.einsum('bchw,bhw->bchw', Y_RB, mask_over) + torch.einsum('bchw,bhw->bchw', Y_G, mask_under)
        # Y_SG = self.gsg(x)
        out = self.final_conv(Y_DI + x)
        return out, mask_over, mask_under

# if __name__ == '__main__':
#     a = torch.randn(1, 3, 512, 512)
#     dec = MSPEC_Net()
#     y =dec(a)
#     print(y.shape)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.cuda.FloatTensor(1, 4, 256, 256).fill_(1)
    model = RawHDR(input_size=256, RB_gudie=True, G_guidance=True, RGB=False).cuda()
    # model = HDRUNet().cuda()
    # model = HDRCNN(device).cuda()
    # model = DCENet().cuda()
    # print(output.shape)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (4, 256, 256), as_strings=True,
                                           print_per_layer_stat=True, verbose=True, ignore_modules=[])
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print(model.gsg.flops())
    output = model(data)
    print(output[0].shape)
