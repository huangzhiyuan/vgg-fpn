from torch.nn.modules.module import Module
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

class MyAdaptiveAvgPool2d(nn.Module):

    def __init__(self, sz=None):
        super(MyAdaptiveAvgPool2d, self).__init__()

    def forward(self, x):
        inp_size = x.size()
        return nn.functional.avg_pool2d(input=x, kernel_size=(inp_size[2], inp_size[3]))

class VGG11bn(nn.Module):

    def __init__(self, init_weights=True):
        super(VGG11bn, self).__init__()
        self.layer0 = self.make_layers(3, [64])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layers(64, [128])
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = self.make_layers(128, [256, 256])
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer3 = self.make_layers(256, [512, 512])
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer4 = self.make_layers(512, [512, 512])
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, inchannels, channels_list, batch_norm=True):
        layers = []
        for v in channels_list:
            conv2d = nn.Conv2d(inchannels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            inchannels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        blocks = []
        x = self.layer0(x)
        x = self.maxpool(x) #2

        x = self.layer1(x)
        x = self.maxpool_1(x) #4
        blocks.append(x)

        x = self.layer2(x)
        x = self.maxpool_2(x) #8
        blocks.append(x)

        x = self.layer3(x)
        x = self.maxpool_3(x) #16
        blocks.append(x)

        x = self.layer4(x)
        x = self.maxpool_4(x) #32
        blocks.append(x)

        return x, blocks

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class East(nn.Module):
    def __init__(self, feat_stride=4, isTrain=True):
        super(East, self).__init__()
        self.isTrain = isTrain
        self.feat_stride = feat_stride
        self.resnet = VGG11bn(True)
        self.conv1 = nn.Conv2d(512+512, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        #self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unpool1 = nn.Upsample(scale_factor=2)
        #self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unpool2 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(128+256, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3 ,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        #self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unpool3 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(64+128, 64, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()
        # self.conv8 = nn.Conv2d(32, 7, 1)

        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.adaptivePool = MyAdaptiveAvgPool2d()

        self.cls_head = self._make_head(2)
        self.link_head = self._make_head(16)
        self.count = 0


    def _make_head(self, out_planes):
        layers = []
        for _ in range(3):
            layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(32, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self,input):
        _, f = self.resnet(input)
        h = f[3]  # bs 2048 w/32 h/32
        g = (self.unpool1(h)) #bs 2048 w/16 h/16
        c = self.conv1(torch.cat((g, f[2]), 1))
        c = self.bn1(c)
        c = self.relu1(c)
        print(c.size())

        h = self.conv2(c)  # bs 128 w/16 h/16
        print(h.size())
        h = self.bn2(h)
        h = self.relu2(h)

        g = self.unpool2(h)  # bs 128 w/8 h/8
        c = self.conv3(torch.cat((g, f[1]), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c)  # bs 64 w/8 h/8
        h = self.bn4(h)
        h = self.relu4(h)

        g = self.unpool3(h) # bs 64 w/4 h/4
        c = self.conv5(torch.cat((g, f[0]), 1))
        c = self.bn5(c)
        c = self.relu5(c)

        h = self.conv6(c) # bs 32 w/4 h/4
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h) # bs 32 w/4 h/4
        g = self.bn7(g)
        g = self.relu7(g)

        cls_preds = self.cls_head(g)
        link_preds = self.link_head(g)

        cls_preds = self.adaptivePool(cls_preds)

        return cls_preds, link_preds
