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
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
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
    elif classname.find("BatchNorm") != -1:
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

        self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
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

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.num_classes + 1), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        if opt.channels_last or opt.device == "cuda":
            out = out.contiguous()
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

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
                prepared_model((fixed_noise, labels))
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

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=True)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=True)
        fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=True)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
