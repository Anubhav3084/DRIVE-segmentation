import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        pool1 = self.pool1(conv1)

        conv2 = F.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        size = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        flattened = pool2.view(-1, size)
        fc1 = F.relu(self.fc1(flattened))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.relu(self.fc3(fc2))

        output = F.softmax(fc3, dim=1)
        return output


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p     # return x for skip connection and p for next layer of conv_block
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c * 2, out_c)
        
    def forward(self, inputs, skip):  # skip: skip connection
        x = self.up(inputs)
#         print(x.shape)
        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:])
        x = torch.cat([x,skip], axis=1) # 2nd axis (axis=1) is the number of channels
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, X=64):
        super().__init__()
        """Encoder"""
        self.e1 = encoder_block(3, X)
        self.e2 = encoder_block(X, 2 * X)
        self.e3 = encoder_block(2 * X, 4 * X)
        self.e4 = encoder_block(4 * X, 8 * X)
        
        """bottleneck"""
        self.b = conv_block(8 * X, 16 * X)
        
        """Decoder"""
        self.d1 = decoder_block(16 * X, 8 * X)
        self.d2 = decoder_block(8 * X, 4 * X)
        self.d3 = decoder_block(4 * X, 2 * X)
        self.d4 = decoder_block(2 * X, X)
        
        """Classifier"""
        self.outputs = nn.Conv2d(X, 1, kernel_size=1, padding=0) # only want 1 channel because it's a binary segmentation problem
        
    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)  # s1: skip connection, p1: pooling output
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        """Bottleneck"""
        b = self.b(p4)

        """Decoder"""
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """Classifier"""
        out = self.outputs(d4)

        return out

if __name__ == '__main__':

    model = UNet().to('cuda')
    x = torch.randn(1, 1, 224, 224).to('cuda')
    print(model(x).shape)

    # print(summary(model, (1, 584, 565)))