"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

try:
    from context_func import context_func
except ModuleNotFoundError as e:
    print("!!!pls check how to add context_func.py from launch_benchmark.sh")
    sys.exit(0)

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--arch", type=str, default="", help="model name")
parser.add_argument('--outf', default='./model', help='folder to output images and model checkpoints')
parser.add_argument('--inference', action='store_true', default=False)
parser.add_argument('--num-warmup', default=10, type=int)
parser.add_argument('--num-iterations', default=100, type=int)
parser.add_argument('--ipex', action='store_true', default=False)
parser.add_argument('--precision', default='float32', help='Precision, "float32" or "bfloat16"')
parser.add_argument('--jit', action='store_true', default=False)
parser.add_argument('--profile', action='store_true', default=False ,help='Trigger profile on current topology.')
parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument('--config_file', type=str, default='./conf.yaml', help='config file for int8 tuning')
parser.add_argument("--quantized_engine", type=str, default=None, help="torch backend quantized engine.")
parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nvFuser')
parser.add_argument('--device', default='xpu', choices=['xpu', 'cuda', 'cpu'], type=str)

opt = parser.parse_args()
print(opt)

# set quantized engine
if opt.quantized_engine is not None:
    torch.backends.quantized.engine = opt.quantized_engine
else:
    opt.quantized_engine = torch.backends.quantized.engine
print("backends quantized engine is {}".format(torch.backends.quantized.engine))

class _DataLoader(object):
    def __init__(self, data=None, batch_size=1):
        self.data = data
        self.batch_size = batch_size
    def __iter__(self):
        yield self.data[0], self.data[1]

cuda = torch.cuda.is_available()
if opt.device == "xpu" or opt.ipex:
    import intel_extension_for_pytorch as ipex

if opt.ipex:
    if opt.precision == "bfloat16":
        # Automatically mix precision
        ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        print("Running with bfloat16...")
    device = ipex.DEVICE
else:
    device = torch.device(opt.device)

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# dataloader = DataLoader(
#     ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
#     batch_size=opt.batch_size,
#     shuffle=True,
#     num_workers=opt.n_cpu,
# )

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def generate(netG, batchsize, device):
    netG.eval()
    n_row = 10
    fixed_noise = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, 3, opt.latent_dim, opt.latent_dim))))
    netG = netG.to(device=device)
    fixed_noise = fixed_noise.to(device=device)
    if opt.channels_last or opt.device == "cuda":
        netG_oob, fixed_noise_oob = netG, fixed_noise
        try:
            netG_oob = netG_oob.to(memory_format=torch.channels_last)
            print("[INFO] Use NHWC model")
            fixed_noise_oob = fixed_noise_oob.to(memory_format=torch.channels_last)
            print("[INFO] Use NHWC input")
        except:
            print("[WARN] Input NHWC failed! Use normal input")
        netG, fixed_noise = netG_oob, fixed_noise_oob

    labels = np.array([num for _ in range(10) for num in range(10)])
    labels = Variable(LongTensor(labels))

    if opt.precision == 'int8':
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization((opt.config_file))
        dataset = (fixed_noise, labels)
        calib_dataloader = _DataLoader(dataset)
        quantizer.calib_dataloader = calib_dataloader
        quantizer.model = common.Model(netG)
        q_model = quantizer()
        netG = q_model.model

    if opt.precision == "fx_int8":
        print('Converting int8 model...')
        qconfig = get_default_qconfig(opt.quantized_engine)
        qconfig_dict = {"": qconfig}
        prepared_model = prepare_fx(netG, qconfig_dict)
        with torch.no_grad():
            for i in range(opt.num_warmup):
                prepared_model(fixed_noise)
        netG = convert_fx(prepared_model)
        print('Convert int8 model done...')

    if opt.jit:
        with torch.no_grad():
            netG = torch.jit.trace(netG, fixed_noise, check_trace=False)
        print("---- Use trace model.")
    if opt.nv_fuser:
        fuser_mode = "fuser2"
    else:
        fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    total_iters = opt.num_warmup + opt.num_iterations
    total_time = 0
    total_sample = 0
    profile_len = total_iters // 2
    with torch.no_grad():
        for i in range(total_iters):
            fixed_noise = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, 3, opt.latent_dim, opt.latent_dim))))
            if opt.channels_last or opt.device == "cuda":
                fixed_noise_oob = fixed_noise
                try:
                    fixed_noise_oob = fixed_noise_oob.to(memory_format=torch.channels_last)
                    print("---- use NHWC input")
                except:
                    print("---- use normal input")
                fixed_noise = fixed_noise_oob
            tic = time.time()
            with context_func(opt.profile if i == profile_len else False, opt.device, fuser_mode) as prof:
                fixed_noise = fixed_noise.to(device=device)
                fake = netG(fixed_noise)
                if opt.device == "xpu":
                    torch.xpu.synchronize()
                elif opt.device == "cuda":
                    torch.cuda.synchronize()
            toc = time.time()
            print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
            if i >= opt.num_warmup:
                total_time += toc -tic
                total_sample += batchsize

    print("Throughput: %.3f image/sec, batchsize: %d, latency = %.2f ms"%(total_sample/total_time, batchsize, total_time/opt.num_iterations*1000))

if opt.inference:
    print("----------------Generation benchmarking---------------")
    if opt.precision == "bfloat16":
        amp_enable = True
        amp_dtype = torch.bfloat16
    elif opt.precision == "float16":
        amp_enable = True
        amp_dtype = torch.float16
    else:
        amp_enable = False
        amp_dtype = torch.float32

    generator = generator.eval()
    if opt.device == "xpu":
        generator = torch.xpu.optimize(model=generator, dtype=amp_dtype)
        print("---- enable xpu optimize")

    with torch.autocast(device_type=opt.device, enabled=amp_enable, dtype=amp_dtype):
        generate(generator, opt.batch_size, device=device)

    import sys
    sys.exit(0)


# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=True)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=True)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
