import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets

class cGAN():
    def __init__(self, n_channels, iH, iW, mini_batch, n_samples, input_dim):
        super(cGAN, self).__init__()
        self.n_channels = n_channels
        self.iH = iH
        self.iW = iW
        self.in_shape = (self.n_channels, self.iH, self.iW)
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.mini_batch = mini_batch
        self.Generator = Generator()
        self.Discriminator = Discriminator()

class Discriminator(cGAN):
    def __init__(self, n_channels, iH, iW, in_shape, mini_batch):
        super().__init__(in_shape, mini_batch)
        self.conv2D_1 = F.conv2d(input = (self.mini_batch, self.n_channels, self.iH, self.iW), stride = 2, padding=0)
        self.leaky_relu_1 = F.leaky_relu()

