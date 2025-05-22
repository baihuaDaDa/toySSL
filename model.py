import torch.nn as nn
from torchvision.models import resnet18, mobilenet_v2, efficientnet_b0, vgg11, regnet_y_8gf, shufflenet_v2_x0_5, vit_b_16

# 简单 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 多层感知机
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# 深度 CNN 模型
class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# LeNet
class TinyLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ResNet-18
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = mobilenet_v2(weights=None)
        base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = efficientnet_b0(weights=None)
        base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)

class RegNetY8GF(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = regnet_y_8gf(weights=None)
        base.stem[0] = nn.Conv2d(1, base.stem[0].out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = shufflenet_v2_x0_5(weights=None)
        base.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)