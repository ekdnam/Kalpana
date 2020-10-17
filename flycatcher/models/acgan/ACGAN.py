from .utils import *
from .generator import Generator
from .discriminator import Discriminator
from .datasets import Dataset

import numpy as np

from torch.autograd import Variable

class ACGAN():
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
        sample_interval: int = 500,
        toShuffle: bool = True
    ):
        super(ACGAN, self).__init__()
        r"""The base model for ACGAN

        Args:
            n_epochs: int. number of epochs for trainig
                default: 200
            batch_size: int. size of batches
                default: 64
            lr: float. the learning rate for adam
                default: 0.0002
            b1: float. decay of first order momentum gradient for adam
                default: 0.5
            b2: float. decay of first order momentum gradient for adam
                default: 0.999
            n_cpu: int. number of cpu threads to use during batch generation
                default: 8
            latent_dim: int. dimensionality of latent space
                default: 100
            n_classes: int. number of classes for dataset
                default: 10
            img_size: int. size of each image dimension
                default: 32
            channels: int. number of image channels
                default: 1
            sample_interval: int. interval between image sampling
                default: 500
            toShuffle: bool. whether to shuffle the dataset
                default: True
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
        self.toShuffle = toShuffle

        """Create Generator"""
        self.generator_block = Generator(self.latent_dim, self.n_classes, self.img_size)

        """Create Discriminator"""
        self.discriminator_block = Discriminator(self.channels, self.n_classes, self.img_size)

        """loss functions"""
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxillary_loss = torch.nn.CrossEntropyLoss()

        """check whether environment has cuda"""
        self.cuda = True if torch.cuda.is_available() else False

        if self.cuda:
            """move all model parameters and buffers to GPU"""
            self.generator_block.cuda()
            self.discriminator_block.cuda()
            self.adversarial_loss.cuda()
            self.auxillary_loss.cuda()
        else:
            """else move everything to CPU"""
            self.generator_block.cpu()
            self.discriminator_block.cpu()
            self.adversarial_loss.cpu()
            self.auxillary_loss.cpu()

        """get dataset"""
        dataset = Dataset(img_size=self.img_size, batch_size=self.batch_size, toShuffle=self.toShuffle)
        self.dataloader = dataset.dataloader()

        """optimizers for the generator and discriminator"""
        self.generator_optimizer = torch.optim.Adam(self.generator_block.parameters(), lr = self.lr, betas = (self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator_block.parameters(), lr = self.lr, betas = (self.b1, self.b2))

        """create tensors"""
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor

    def train(self):
        r"""
        train the model
        """
        for epoch in range(self.n_epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):

                batch_size = imgs.shape[0]

                """adversarial ground truths"""
                valid = Variable(self.FloatTensor(self.batch_size, 1).fill_(1.0), requires_grad = False)
                fake = Variable(self.FloatTensor(self.batch_size, 1).fill_(0.0), requires_grad = False)

                """configure input"""
                real_imgs = Variable(imgs.type(self.FloatTensor))
                labels = Variable(labels.type(self.LongTensor))

                """train the generator"""
                self.generator_optimizer.zero_grad()

                """sample noise and labels as generator input"""
                z = Variable(
                    self.FloatTensor(
                        np.random.normal(0.1, (self.batch_size, self.latent_dim))
                        )
                    )
                gen_labels = Variable(
                    self.LongTensor(
                        np.random.randint(0, self.n_classes, self.batch_size)
                        )
                    )
                
                """generate a batch of images"""

                gen_imgs = self.generator_block(z, gen_labels)

                """measure generator's ability to fool discriminator"""
                validity, pred_label = self.discriminator_block(gen_imgs)
                gen_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxillary_loss(pred_label, gen_labels))

                gen_loss.backward()
                self.generator_optimizer.step()


                """train discriminator"""
                self.discriminator_optimizer.zero_grad()

                """loss for real images"""
                real_pred, real_aux = self.discriminator_block(real_imgs)
                d_real_loss = (self.adversarial_loss(real_pred, valid)+ self.auxillary_loss(real_aux, labels)) / 2

                """loss for fake images"""
                fake_pred, fake_aux = self.discriminator_block(gen_imgs.detach())
                d_fake_loss = (self.adversarial_loss(fake_pred, fake)+ self.auxillary_loss(fake_aux, labels)) / 2

                """total discriminator loss"""
                d_loss = (d_fake_loss + d_real_loss) / 2

                """calculate discriminator accuracy"""
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                self.discriminator_optimizer.backward()
                
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch, self.n_epochs, i, len(self.dataloader), d_loss.item(), 100 * d_acc, gen_loss.item())
                )
                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:
                    sample_image(n_row=10, batches_done=batches_done, self.FloatTensor, self.LongTensor)