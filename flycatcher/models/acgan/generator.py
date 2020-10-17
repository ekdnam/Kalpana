import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size, n_channels):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)
        self.n_channels = n_channels
        self.init_size = img_size
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.n_channels, 3, stride=1, padding=1)
        )

        def forward(self, noise, labels):
            gen_input = torch.mul(self.label_emb(labels), noise)
            out = self.l1(gen_input)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.model(out)
            return img