import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetSSD_Orig(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetSSD_Orig, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layer6 = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        # nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
                                        # nn.BatchNorm2d(1024),
                                        # nn.ReLU(inplace=True),

                                        # nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1),
                                        # nn.BatchNorm2d(1024),
                                        # nn.ReLU(inplace=True),

                                    )
        # self.new_layer1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
        #                                 nn.BatchNorm2d(256),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        #                                 nn.BatchNorm2d(512),
        #                                 nn.ReLU(inplace=True))
        #
        # self.new_layer2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
        #                                 nn.BatchNorm2d(128),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
        #                                 nn.BatchNorm2d(128),
        #                                 nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.conv5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # c2 = x
        x = self.layer2(x)
        c3 = self.conv3(x)

        x = self.layer3(x)
        c4 = self.conv4(x)

        x = self.layer4(x)
        # x = self.new_layer1(x)
        c5 = self.conv5(x)

        x = self.layer6(x)
        # x = self.new_layer2(x)
        c6 = x


        return [c3,c4,c5,c6]

def resnetssd101_Orig(cfg):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD_Orig(Bottleneck, [3, 4, 23, 3])

    return model

class conv1xk(nn.Module):
    def __init__(self, c, k):
        super(conv1xk,self).__init__()
        self.conv0 = nn.Conv2d(c, 256, kernel_size=(1, 1), stride=1, bias=False)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=(1, k), stride=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(k, 1), stride=1, bias=False)
        self.conv11 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.conv12 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        # self.conv00 = nn.Conv2d(256, c, kernel_size=(1, 1), stride=1, bias=False)
        # self.conv1 = nn.Conv2d(c, c//2, kernel_size=(1, k), stride=1, bias=False)
        # self.conv2 = nn.Conv2d(c, c//2, kernel_size=(k, 1), stride=1, bias=False)
        # self.conv11 = nn.Conv2d(c, c//2, kernel_size=1, stride=1, bias=False)
        # self.conv12 = nn.Conv2d(c, c//2, kernel_size=1, stride=1, bias=False)
        # self.normal = nn.BatchNorm2d(256)

    def fill_edge(self,x1):
        if x1.size()[2] < x1.size()[3]:
            fill = torch.full((x1.size()[0], x1.size()[1], (x1.size()[3]-x1.size()[2])//2, x1.size()[3]), 0).cuda()
            return torch.cat((fill, x1, fill), dim=2)
        if x1.size()[2] > x1.size()[3]:
            fill = torch.full((x1.size()[0], x1.size()[1], x1.size()[2], (x1.size()[2]-x1.size()[3])//2), 0).cuda()
            return torch.cat((fill, x1, fill), dim=3)

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.fill_edge(self.conv1(x))
        # x1 = F.relu(x1)
        x2 = self.fill_edge(self.conv2(x))
        # x2 = F.relu(x2)

        # y = F.relu(self.conv3(x1 + x2))
        y = F.relu(self.conv11(torch.cat((x1,x2), dim=1)))
        # y = F.relu(torch.cat((x1, x2), dim=1))
        x = F.relu(self.conv12(x))
        return torch.cat((x, y), dim=1)
        # return self.normal(torch.cat((x, y), dim=1))
        # return self.normal(self.conv00(torch.cat((x, y), dim=1)))
        # return y

class ResNetSSD_1xk(nn.Module):

    def __init__(self, layers=[3, 4, 23, 3], num_classes=2):
        self.inplanes = 64
        super(ResNetSSD_1xk, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)
        self.layer6 = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True))
        # self.new_layer1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
        #                                 nn.BatchNorm2d(256),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        #                                 nn.BatchNorm2d(512),
        #                                 nn.ReLU(inplace=True))
        #
        # self.new_layer2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
        #                                 nn.BatchNorm2d(128),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #                                 nn.BatchNorm2d(256),
        #                                 nn.ReLU(inplace=True))



        self.num_classes = num_classes


        # self.conv1xk_c1 = conv1xk(64, 3)
        # self.conv1xk_c2 = conv1xk(256, 3)
        self.conv1xk_c3 = conv1xk(512, 3)
        self.conv1xk_c4 = conv1xk(1024, 3)
        self.conv1xk_c5 = conv1xk(2048, 3)
        # self.conv1xk_c5 = conv1xk(512, 3)
        self.conv1xk_c6 = conv1xk(256, 3)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # att1 = self.conv1xk_c1(x)
        # x = x + att1
        x = self.maxpool(x)

        x  = self.layer1(x)
        # x = self.conv1xk_c2(x)
        # x = x + att2

        x  = self.layer2(x)
        c3 = self.conv1xk_c3(x)
        #
        # x = self.conv1xk_c3(x)
        # x = x + att3
        # c2 = x

        x  = self.layer3(x)
        c4 = self.conv1xk_c4(x)

        # x = self.conv1xk_c4(x)
        # x = x +att4
        # c4 = x

        x  = self.layer4(x)
        # x = self.new_layer1(x)
        c5 = self.conv1xk_c5(x)

        # x = self.conv1xk_c5(x)
        # x = x + att5
        # c5 = x

        x  = self.layer6(x)
        # x = self.new_layer2(x)
        c6 = self.conv1xk_c6(x)
        #
        # x = self.conv1xk_c6(x)
        # x = x + att6
        # c6 = x


        return [c3,c4,c5,c6]

