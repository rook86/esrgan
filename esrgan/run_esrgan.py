import numpy as np
import random
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from datasets import *
from discriminator import *
from generator import *
from losses import *

#===variable===#
block = 10
learning_rate = 0.0001
sample = 50
num_epoch = 10
data_path = "../sr_datasets/DIV2K_train_HR"

os.makedirs("train_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

cuda = torch.cuda.is_available()
Device = 'cuda' if torch.cuda.is_available() else 'cpu'

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../sr_datasets/DIV2K_train_HR", hr_shape=(512,512)),
    batch_size=1,
    shuffle=True,
    num_workers=8,
)


criterion_MSE = torch.nn.MSELoss()

if cuda:
    criterion_MSE = criterion_MSE.cuda()
    criterion_pe = PerceptualLoss().cuda()
    criterion_gan = GANLoss().cuda()
    generator = Generator(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=block, num_grow_ch=32).cuda()
    discriminator = Discriminator().cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.99))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.99))


#===train===#
for epoch in range(0,num_epoch):
    for i, img in enumerate(dataloader):
        

        # Configure model input
        imgs_lr = Variable(img["lr"].type(Tensor))
        imgs_hr = Variable(img["hr"].type(Tensor))

        chunk_dim = 4
        a_x_split = torch.chunk(imgs_lr, chunk_dim, dim=2)

        chunks_lr = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_lr.append(c_)
        
        a_x_split = torch.chunk(imgs_hr, chunk_dim, dim=2)

        chunks_hr = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_hr.append(c_)
        
        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        total_loss_G = 0
        gen_hrs = []
        for j in range(len(chunks_lr)):
            gen_hrs.append(generator(chunks_lr[j]))
            fake_pred = discriminator(generator(chunks_lr[j]))
            loss_GAN = criterion_gan(fake_pred, True, is_disc=False)
            loss_content = criterion_pe(generator(chunks_lr[j]), chunks_hr[j])
            loss_pixel = criterion_MSE(generator(chunks_lr[j]), chunks_hr[j])
            # Total loss
            loss_G = loss_GAN + loss_pixel + loss_content
            total_loss_G += loss_G
            loss_G.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        total_loss_D = 0
        for j in range(len(chunks_lr)):
            real_pred = discriminator(chunks_hr[j])
            fake_pred = discriminator(generator(chunks_lr[j]))
            loss_real = criterion_gan(real_pred, True, is_disc=True)
            loss_fake = criterion_gan(fake_pred, False, is_disc=True)
            # Total loss
            loss_D = loss_real + loss_fake
            total_loss_D += loss_D
            loss_D.backward()
            optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (
                epoch,
                i,
                len(dataloader),
                total_loss_D/16,
                total_loss_G/16
            )
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % 50 == 0:
            h1 = torch.cat((gen_hrs[0],gen_hrs[1],gen_hrs[2],gen_hrs[3]),3)
            h2 = torch.cat((gen_hrs[4],gen_hrs[5],gen_hrs[6],gen_hrs[7]),3)
            h3 = torch.cat((gen_hrs[8],gen_hrs[9],gen_hrs[10],gen_hrs[11]),3)
            h4 = torch.cat((gen_hrs[12],gen_hrs[13],gen_hrs[14],gen_hrs[15]),3)
            h12 = torch.cat((h1,h2),2)
            h34 = torch.cat((h3,h4),2)
            gen_hr = torch.cat((h12,h34),2)
            PSNR = 10 * math.log(criterion_MSE(imgs_hr, gen_hr),10)
            print("[PSNR %f]" % (PSNR))
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr),-1)
            save_image(img_grid, "train_images/%d.png" % batches_done)

        
#===save_model===#
torch.save(generator.state_dict(), "saved_models/generator.pth")
torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")
