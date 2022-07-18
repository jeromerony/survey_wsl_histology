import sys
from os.path import dirname, abspath


root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['InceptionV3']

'https://github.com/clovaai/wsolevaluation/blob/master/wsol/inception.py'

pretrained_settings = {
    'inceptionv3': {
        'imagenet': {
            'url': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        }
    }
}


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=False)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, 1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, 1)
        self.branch5x5_2 = BasicConv2d(48, 64, 5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size,
                                     stride=stride, padding=padding)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3,
                                          stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride,
                                   padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, 1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7_2 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, (7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, (1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super(InceptionV3, self).__init__()

        self.large_feature_map = True  # default. out of 224*224 is 28*28.

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(768, 1024, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=False),
        )

        self.features = nn.Sequential(
            self.Conv2d_1a_3x3,
            self.Conv2d_2a_3x3,
            self.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            self.Conv2d_3b_1x1,
            self.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            self.Mixed_5b,
            self.Mixed_5c,
            self.Mixed_5d,
            self.Mixed_6a,
            self.Mixed_6b,
            self.Mixed_6c,
            self.Mixed_6d,
            self.Mixed_6e,
            self.SPG_A3_1b,
            self.SPG_A3_2b
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = None

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x):
        feat_map = self.features(x)
        print(feat_map.shape)
        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        return logits


def inceptionv3(num_classes=1000, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['inceptionv3'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV3(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionV3(num_classes=num_classes)
    return model


def test_inception3():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from torchvision.models.inception import inception_v3

    model = InceptionV3(num_classes=100)
    model.to(device)
    bsize = 2
    x = torch.rand(bsize, 3, 224, 224).to(device)
    labels = torch.zeros((bsize,), dtype=torch.long)
    out = model(x)
    print(x.shape, out.shape)


if __name__ == "__main__":
    test_inception3()
