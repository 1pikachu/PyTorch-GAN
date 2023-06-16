import argparse
import os
import numpy as np
import math
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

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

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
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
parser.add_argument('--config_file', type=str, default='./conf.yaml', help='config file for int8 tuning')
parser.add_argument("--quantized_engine", type=str, default=None, help="torch backend quantized engine.")
parser.add_argument("--nv_fuser", action='store_true', default=False, help='enable nvFuser')
parser.add_argument('--device', default='xpu', choices=['xpu', 'cuda', 'cpu'], type=str)
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
if opt.device == "xpu" or opt.ipex:
    import intel_extension_for_pytorch as ipex

if opt.ipex:
    if opt.precision == "bfloat16":
        # Automatically mix precision
        ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        print("Running with bfloat16...")
    device = ipex.DEVICE
else:
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(opt.device)

# set quantized engine
if opt.quantized_engine is not None:
    torch.backends.quantized.engine = opt.quantized_engine
else:
    opt.quantized_engine = torch.backends.quantized.engine
print("backends quantized engine is {}".format(torch.backends.quantized.engine))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class _DataLoader(object):
    def __init__(self, data=None, batch_size=1):
        self.data = data
        self.batch_size = batch_size
    def __iter__(self):
        yield self.data[0], self.data[1]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        if opt.channels_last or opt.device == "cuda":
            out = out.to(memory_format=torch.channels_last)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        if opt.channels_last or opt.device == "cuda":
            out = out.contiguous()
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        if opt.channels_last or opt.device == "cuda":
            out = out.to(memory_format=torch.channels_last)
        return out


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def generate(netG, batchsize, device):
    netG.eval()
    fixed_noise = Variable(FloatTensor(np.random.normal(0, 1, (10 ** 2, opt.latent_dim))))
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
    profile_len = total_iters//2
    with torch.no_grad():
        for i in range(total_iters):
            fixed_noise = Variable(FloatTensor(np.random.normal(0, 1, (10 ** 2, opt.latent_dim))))
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

# BEGAN hyper parameters
gamma = 0.75
lambda_k = 0.001
k = 0.0

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs.detach())

        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
        d_loss = d_loss_real - k * d_loss_fake

        d_loss.backward()
        optimizer_D.step()

        # ----------------
        # Update weights
        # ----------------

        diff = torch.mean(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff)).data[0]

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
