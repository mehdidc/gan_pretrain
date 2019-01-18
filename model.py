import numpy as np
from itertools import chain

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform


class Gen(nn.Module):
    def __init__(
        self,
        latent_size=100,
        nb_gen_filters=64,
        nb_colors=1,
        image_size=64,
        act="tanh",
        nb_extra_layers=0,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.nb_gen_filters = nb_gen_filters
        self.nb_colors = nb_colors
        self.image_size = image_size
        self.act = act

        nz = self.latent_size
        ngf = self.nb_gen_filters
        w = self.image_size
        nc = self.nb_colors

        nb_blocks = int(np.log(w) / np.log(2)) - 3
        nf = ngf * 2 ** (nb_blocks + 1)
        if nb_blocks >= 0:
            layers = [
                nn.ConvTranspose2d(nz, nf, 4, 1, 0, bias=False),
                nn.BatchNorm2d(nf),
                nn.ReLU(True),
            ]
        else:
            layers = [
                nn.ConvTranspose2d(nz, nf, 2, 1, 0, bias=False),
                nn.BatchNorm2d(nf),
                nn.ReLU(True),
            ]
        for _ in range(nb_blocks):
            layers.extend(
                [
                    nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(nf // 2),
                    nn.ReLU(True),
                ]
            )
            nf = nf // 2
        for _ in range(nb_extra_layers):
            layers.extend(
                [
                    nn.ConvTranspose2d(nf, nf, 1, 1, bias=False),
                    nn.BatchNorm2d(nf),
                    nn.ReLU(True),
                ]
            )
        layers.append(nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=True))
        layers.append(nn.Tanh() if act == "tanh" else nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        out = self.main(input)
        return out


class Discr(nn.Module):
    def __init__(
        self, nb_colors=1, nb_discr_filters=64, image_size=64, nb_extra_layers=0
    ):
        super().__init__()

        self.nb_colors = nb_colors
        self.nb_discr_filters = nb_discr_filters
        self.image_size = image_size

        w = self.image_size
        ndf = self.nb_discr_filters
        nc = self.nb_colors

        nb_blocks = int(np.log(w) / np.log(2)) - 3
        nf = ndf
        if nb_blocks >= 0:
            layers = [
                nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            for _ in range(nb_blocks):
                layers.extend(
                    [
                        nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(nf * 2),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
                nf = nf * 2
        else:
            layers = [
                nn.Conv2d(nc, nf, 1, 1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        for _ in range(nb_extra_layers):
            layers.extend(
                [nn.Conv2d(nf, nf, 1, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(True)]
            )
        layers.append(nn.Conv2d(nf, 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        out = self.main(input)
        return out.view(out.size(0), 1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == "Linear":
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)
