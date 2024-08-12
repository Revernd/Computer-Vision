import torch
import torch.nn as nn

class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.conv1 = conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # in_channels, out_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool
        self.inception_3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.auxiliary1 = auxiliary_block(512, num_classes)

        self.inception_4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.auxiliary2 = auxiliary_block(528, num_classes)

        self.inception_4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)
        self.sofmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool3(x)

        x = self.inception_4a(x)
        aux1 = self.auxiliary1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        aux2 = self.auxiliary2(x)
        
        x = self.inception_4e(x)    
        x = self.max_pool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.sofmax(self.fc1(x))

        return x, aux1, aux2




class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1_pool, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv(x)))
        return x
    
class auxiliary_block(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(auxiliary_block, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)  # Adjusted to match PyTorch convention
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    model = GoogleNet()
    print([x.shape for x in model(x)])