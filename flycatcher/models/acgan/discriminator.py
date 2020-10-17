import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels: int, n_classes: int, img_size: int):
        r"""The discriminator block of a GAN
        Corrects the Generator

        Args:
            channels: int. Number of channels the image has
            n_classes: int. Number of classes for the dataset
            img_size: int. Size of the image
        """
        super(Discriminator, self).__init__()

        self.channels = channels
        self.n_classes = n_classes
        self.img_size = img_size

        def discriminator_block(in_filters: int, out_filters: int, apply_batch_norm: bool = True):
            r"""Return a layer of each discriminator block"""
            block = [
                nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=3, stride=2, padding=1), 
                nn.LeakyReLU(negative_slope=0.2,inplace=True),
                nn.Dropout2d(0.25)
            ]
            if apply_batch_norm is True:
                block.append(nn.BatchNorm2d(num_features=out_filters, eps=0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_filters=self.channels, out_filters=16, apply_batch_norm=False)
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        """size of the downsampled image"""
        self.downsampled_image_size = img_size//2**4

        """output layer 1"""
        self.adv_layer = nn.Sequential(
            nn.Linear(in_features=128*self.downsampled_image_size**2, out_features=1),
            nn.Sigmoid()
        )

        """output layer 2"""
        self.aux_layer = nn.Sequential(
            nn.Linear(in_features=128*self.downsampled_image_size**2, out_features=self.n_classes),
            nn.Softmax()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label