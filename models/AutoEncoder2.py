import torch
import torch.nn.functional as F

class Concatenation(torch.nn.Module):
    def __init__(self):
        super(Concatenation, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = F.leaky_relu(input=self.conv1(x), negative_slope=0.01, inplace=False)
        x = F.leaky_relu(input=self.conv2(x), negative_slope=0.01, inplace=False)
        x = F.leaky_relu(input=self.conv3(x), negative_slope=0.01, inplace=False)
        x = F.leaky_relu(input=self.conv4(x), negative_slope=0.01, inplace=False)
        x = F.leaky_relu(input=self.conv5(x), negative_slope=0.01, inplace=False)
        x = F.leaky_relu(input=self.conv6(x), negative_slope=0.01, inplace=False)
        x = torch.nn.functional.interpolate(input=x, size=(88, 88), mode='bilinear', align_corners=False)
        x = torch.nn.functional.interpolate(input=x, size=(176, 176), mode='bilinear', align_corners=False)
        return x