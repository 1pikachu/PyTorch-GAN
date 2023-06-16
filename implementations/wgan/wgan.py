import argparse
import os
import numpy as np
import math
import sys
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
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
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
    device = torch.device(opt.device)

class _DataLoader(object):
    def __init__(self, data=None, batch_size=1):
        self.data = data
        self.batch_size = batch_size
    def __iter__(self):
        yield self.data[0], self.data[1]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator.to(opt.device)
discriminator.to(opt.device)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

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
    profile_len = total_iters // 2
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

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1
