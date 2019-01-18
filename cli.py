import os
import shutil
from functools import partial
import numpy as np
from clize import run

from skimage.io import imsave

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from viz import grid_of_images_default

from model import Gen
from model import Discr

from data import load_dataset, PatchDataset


def save_weights(m, folder="out", prefix=""):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        if np.sqrt(w.size(1)) == int(w.size(1)):
            s = int(np.sqrst(w.size(1)))
            w = w.view(w.size(0), 1, s, s)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave("{}/{}_feat_{}.png".format(folder, prefix, w.size(0)), gr)
    elif isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        if w.size(1) == 1:
            w = w.view(w.size(0) * w.size(1), w.size(2), w.size(3))
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave("{}/{}_feat_{}.png".format(folder, prefix, w.size(0)), gr)
        elif w.size(1) == 3:
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave("{}/{}_feat_{}.png".format(folder, prefix, w.size(0)), gr)


def train(
    *,
    folder="out",
    dataset="mnist",
    image_size=None,
    resume=False,
    wasserstein=False,
    log_interval=1,
    device="cpu",
    batch_size=64,
    nz=100,
    parent_model=None,
    freeze_parent=True,
    num_workers=1,
    nb_filters=64,
    nb_epochs=200,
    nb_extra_layers=0,
    nb_draw_layers=1
):

    try:
        os.makedirs(folder)
    except Exception:
        pass
    lr = 0.0002
    dataset = load_dataset(dataset, split="train", image_size=image_size)
    x0, _ = dataset[0]
    nc = x0.size(0)
    w = x0.size(1)
    h = x0.size(2)
    _save_weights = partial(save_weights, folder=folder, prefix="gan")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    act = "sigmoid" if nc == 1 else "tanh"
    if resume:
        gen = torch.load("{}/gen.th".format(folder))
        discr = torch.load("{}/discr.th".format(folder))
    else:
        gen = Gen(
            latent_size=nz,
            nb_colors=nc,
            image_size=w,
            act=act,
            nb_gen_filters=nb_filters,
            nb_extra_layers=nb_extra_layers,
        )
        discr = Discr(
            nb_colors=nc,
            image_size=w,
            nb_discr_filters=nb_filters,
            nb_extra_layers=nb_extra_layers,
        )
    print(gen)
    print(discr)
    if wasserstein:
        gen_opt = optim.RMSprop(gen.parameters(), lr=lr)
        discr_opt = optim.RMSprop(discr.parameters(), lr=lr)
    else:
        gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
        discr_opt = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))

    input = torch.FloatTensor(batch_size, nc, w, h)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    label = torch.FloatTensor(batch_size)

    if wasserstein:
        real_label = 1
        fake_label = -1

        def criterion(output, label):
            return (output * label).mean()

    else:
        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()

    gen = gen.to(device)
    discr = discr.to(device)
    input, label = input.to(device), label.to(device)
    noise = noise.to(device)

    giter = 0
    diter = 0

    dreal_list = []
    dfake_list = []
    pred_error_list = []

    for epoch in range(nb_epochs):
        for i, (X, _) in enumerate(dataloader):
            if wasserstein:
                # clamp parameters to a cube
                for p in discr.parameters():
                    p.data.clamp_(-0.01, 0.01)
            # Update discriminator
            discr.zero_grad()
            batch_size = X.size(0)
            X = X.to(device)
            input.resize_as_(X).copy_(X)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)
            output = discr(inputv)
            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])
            errD_real = criterion(labelpred, labelv)
            errD_real.backward()
            D_x = labelpred.data.mean()
            dreal_list.append(D_x)
            noise.resize_(batch_size, nz, 1, 1).uniform_(-1, 1)
            noisev = Variable(noise)
            fake = gen(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = discr(fake.detach())

            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])
            errD_fake = criterion(labelpred, labelv)
            errD_fake.backward()
            D_G_z1 = labelpred.data.mean()
            dfake_list.append(D_G_z1)
            discr_opt.step()
            diter += 1

            # Update generator
            gen.zero_grad()
            fake = gen(noisev)
            labelv = Variable(label.fill_(real_label))
            output = discr(fake)
            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])
            errG = criterion(labelpred, labelv)
            errG.backward()
            gen_opt.step()
            if diter % log_interval == 0:
                print(
                    "{}/{} dreal : {:.6f} dfake : {:.6f}".format(
                        epoch, nb_epochs, D_x, D_G_z1
                    )
                )
            if giter % 100 == 0:
                x = 0.5 * (X + 1) if act == "tanh" else X
                f = 0.5 * (fake.data + 1) if act == "tanh" else fake.data
                vutils.save_image(
                    x, "{}/real_samples_last.png".format(folder), normalize=True
                )
                vutils.save_image(
                    f,
                    "{}/fake_samples_epoch_{:03d}.png".format(folder, epoch),
                    normalize=True,
                )
                vutils.save_image(
                    f, "{}/fake_samples_last.png".format(folder), normalize=True
                )
                torch.save(gen, "{}/gen.th".format(folder))
                torch.save(discr, "{}/discr.th".format(folder))
                gen.apply(_save_weights)
            giter += 1

if __name__ == "__main__":
    run([train])
