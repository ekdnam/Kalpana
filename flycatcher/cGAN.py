import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

class Opt:
    def __init__(
        self,
        n_epochs: int = 200,
        batch_size: int = 64,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        n_cpu: int = 8,
        latent_dim: int = 100,
        n_classes: int = 10,
        img_size: int = 32,
        channels: int = 1,
        sample_interval: int = 400,
    ):
        r"""A class which contains all the required inputs for the model

        n_epochs: number of epochs of training

        batch_size: size of batches

        lr: adam learning rate

        b1: adam decay of first order momentum of gradient

        b2: adam decay of first order momentum of gradient

        n_cpu: number of cpu threads during batch generation

        latent_dim: dimensionality of latent space

        n_classes: number of classes for dataset

        img_size: size of each image channels

        channels: number of image channels

        sample_interval: interval between image sampling
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.n_cpu = n_cpu
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_size = img_size
        self.channels = channels
        self.sample_interval = sample_interval


opt = Opt()
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

adversarial_loss = torch.nn.MSELoss()

generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

os.makedirs("../../data/mnist", exist_ok = True)
datalaoder = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,

    )
)