from torch.utils.data import DataLoader
from torchvision import datasets

import os

import torchvision.transforms as transforms

class Dataset():
    def __init__(self, img_size: int = 32, batch_size: int = 64, toShuffle: bool = True):
        r"""Creates a dataset

        The dataset is going to be used to train the GAN.

        Args:
            img_size(int): the size of the image. this is going to be used to transform the image
                default: 32
            batch_size(int): the size of the batches
                default: 64
            toShuffle(bool): whether to shuffle the data
                default: True
        """
        super(Dataset, self).__init__()
        os.makedirs("../../data/mnist", exist_ok = True)
        self.img_size = img_size
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            datasets.MNIST(
                "../../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size = self.batch_size,
            shuffle=toShuffle
        )