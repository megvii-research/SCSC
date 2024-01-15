import copy
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

__all__ = ['CBTNet', 'cbtnet50', 'cbtnet101', 'cbtnet152']

def channel_shuffle_v2(x, m):
    batchsize, num_channels, height, width = x.data.size()

    # reshape
    x = x.reshape(batchsize * num_channels // m,
        m, height * width)
    x = x.permute(1, 0, 2)
    # flatten
    x = x.reshape(m, batchsize, num_channels // m, height, width)

    xs = [x[i] for i in range(m)]

    return torch.cat(xs, dim=1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channel = max(channel // reduction, 32)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNN_Trans_block(nn.Module):

    expansion = 3

    def __init__(self, in_planes, out_planes, kernel_sizes, strides, paddings, nums, hw, mg, dilation=1, downsample = None):
        super().__init__()
        self.m = len(kernel_sizes)
        self.mg = mg
        self.strides = strides
        self.chunk_planes = in_planes//self.m

        self.reduction_c = nn.Conv2d(in_planes, self.chunk_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.chunk_planes)
        self.act1 = nn.GELU()

        self.layers = nn.ModuleList()
        for kernel_size, stride, padding, num in zip(kernel_sizes, strides, paddings, nums):
            self.layers.append(nn.ModuleList([nn.Conv2d(self.chunk_planes, self.chunk_planes, kernel_size, stride, padding, groups=self.chunk_planes, bias=False, dilation=dilation) for _ in range(num)]))
        self.bn2 = nn.BatchNorm2d(self.chunk_planes*self.m)
        self.act2 = nn.GELU()

        self.recover_c = nn.Conv2d(self.chunk_planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.act3 = nn.GELU()

        self.routing_fc = nn.Conv2d(self.chunk_planes, self.m*self.mg, kernel_size=1, stride=strides[0], bias=False)

        self.downsample = downsample

        self.se = SELayer(out_planes)

    def forward(self, input):
        n, c, h, w = 0, 0, 0, 0
        if self.strides[0]>1:
            n,c,h,w = input.shape[0], input.shape[1], input.shape[2]//2, input.shape[3]//2
        else:
            n, c, h, w = input.shape
        identity = input

        x = self.reduction_c(input)
        x = self.bn1(x)
        x_reduc = self.act1(x)

        #print(self._get_name(), x.shape)

        tmp_xs = []
        for i, patch_size in enumerate(self.patch_sizes):
            if len(self.layers[i])==1:
                tmp_x = self.layers[i][0](x_reduc)
            else:
                tmp_x = list(map(lambda conv: conv(x_reduc).reshape(n, -1, 1), self.layers[i]))
                tmp_x = torch.cat(tmp_x, dim=2)
                tmp_x = F.fold(tmp_x, output_size=(h, w), kernel_size=patch_size, stride=patch_size)
            tmp_xs.append(tmp_x)
        x = channel_shuffle_v2(torch.cat(tmp_xs, dim=1), self.m)

        x = self.bn2(x)
        x = self.act2(x)
        x = x.reshape(n, self.m, self.mg, self.chunk_planes//self.mg, h*w).permute(0,4,2,3,1).contiguous()

        routing_x = F.sigmoid(self.routing_fc(x_reduc))
        routing_x = routing_x.reshape(n, self.m, self.mg, h*w, 1).permute(0,3,2,1,4).contiguous()

        x = torch.matmul(x, routing_x)
        x = x.reshape(n, h, w, self.chunk_planes).permute(0,3,1,2).contiguous()

        x = self.recover_c(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.se(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return x+identity


class CBTNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[CNN_Trans_block]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        replace_stride_with_dilation: Optional[List[bool]] = None
    ) -> None:
        super(CBTNet, self).__init__()

        self.inplanes = 96
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 96, layers[0], stride=2, kernel_sizes=[3, 9, 13], strides=[1,1,1], paddings=[1,4,6], nums=[1,1,1], hw=(112,112), mg=4)
        self.layer2 = self._make_layer(block, 192, layers[1], stride=2, kernel_sizes=[3, 7, 11], strides=[1,1,1], paddings=[1,3,5], nums=[1,1,1], hw=(56,56), mg=4, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 384, layers[2], stride=2, kernel_sizes=[3, 5, 7], strides=[1,1,1], paddings=[1,2,3], nums=[1,1,1], hw=(28,28), mg=4, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, kernel_sizes=[3, 5], strides=[1,1], paddings=[1,2], nums=[1,1], hw=(14,14), mg=4, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, CNN_Trans_block):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[Union[CNN_Trans_block]], planes: int, blocks: int,
                    stride: int, kernel_sizes, strides, paddings, nums, hw, mg, dilate: bool = False) -> nn.Sequential:
        downsample = None
        down_kernel_sizes = copy.deepcopy(kernel_sizes)
        down_strides = copy.deepcopy(strides)
        down_paddings = copy.deepcopy(paddings)
        down_nums = copy.deepcopy(nums)
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        #print('strides', strides)
        if stride != 1 or self.inplanes != planes * block.expansion:
            # down_kernel_sizes = kernel_sizes[:-1]
            # down_strides = strides[:-1]
            # down_paddings = paddings[:-1]
            # down_nums = nums[:-1]
            for i in range(len(down_strides)):
                if down_strides[i]==1:
                    down_strides[i]=2
                if down_nums[i] % 4 ==0:
                    down_nums[i] = down_nums[i]//4
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        #print('strides2', strides)
        layers = []
        layers.append(block(self.inplanes, planes * block.expansion, down_kernel_sizes, down_strides, down_paddings, down_nums, hw, mg, dilation=previous_dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes * block.expansion, kernel_sizes, strides, paddings, nums, hw=(hw[0]//2, hw[1]//2), mg=mg, dilation=self.dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def cbtnet50(**kwargs: Any) -> CBTNet:
    return CBTNet(CNN_Trans_block, [3, 5, 12, 3], **kwargs)


def cbtnet101(**kwargs: Any) -> CBTNet:
    return CBTNet(CNN_Trans_block, [3, 4, 23, 3], **kwargs)


def cbtnet152(**kwargs: Any) -> CBTNet:
    return CBTNet(CNN_Trans_block, [3, 8, 36, 3], **kwargs)
