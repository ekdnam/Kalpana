import torch
from torch.autograd import Variable

import numpy as np

from torchvision.utils import save_image


def initialize_weights_normally(m):
    classname = m.__clase.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def save_image(
    n_row: int, batches_done: int, latent_dim: int, FloatTensor, LongTensor, generator
):
    r"""Save a grid of generated digits ranging from 0 to n_classes"""

    """Sample noise"""
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))

    """Get labels ranging from 0 to n_classes for n_rows"""
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(
        gen_imgs.data, "output_images/acgan/images/%d.png" % batches_done, nrow=n_row, normalize=True
    )
