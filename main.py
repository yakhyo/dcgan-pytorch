import os
import yaml
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from nets.nn import Generator, Discriminator

# load config.yaml
with open(r'config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# make assets dir if does not exist
try:
    os.makedirs(config['assets'])
except OSError:
    pass

print("Random Seed: ", config['manualSeed'])
random.seed(config['manualSeed'])
torch.manual_seed(config['manualSeed'])

if config['dataroot'] is None and str(config['dataset']).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % config['dataset'])

if config['dataset'] in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = torchvision.datasets.ImageFolder(root=config['dataroot'],
                                               transform=transforms.Compose([
                                                   transforms.Resize(config['image_size']),
                                                   transforms.CenterCrop(config['image_size']),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
    nc = 3
elif config['dataset'] == 'lsun':
    classes = [c + '_train' for c in config.classes.split(',')]
    dataset = torchvision.datasets.LSUN(root=config.dataroot, classes=classes,
                                        transform=transforms.Compose([
                                            transforms.Resize(config['image_size']),
                                            transforms.CenterCrop(config['image_size']),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
    nc = 3
elif config['dataset'] == 'cifar10':
    dataset = torchvision.datasets.CIFAR10(root=config.dataroot, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(config['image_size']),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
    nc = 3

elif config['dataset'] == 'mnist':
    dataset = torchvision.datasets.MNIST(root=config['dataroot'], download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(config['image_size']),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,)),
                                         ]))
    nc = 1

elif config['dataset'] == 'fake':
    dataset = torchvision.datasets.FakeData(image_size=(3, config['image_size'], config['image_size']),
                                            transform=transforms.ToTensor())
    nc = 3

assert dataset
assert nc

data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
                                          shuffle=True, num_workers=int(config['workers']))

# hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpu = int(config['ngpu'])
nz = int(config['nz'])
ngf = int(config['ngf'])
ndf = int(config['ndf'])


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# generator
netG = Generator(nz, ngf, nc, num_gpu).to(device)
netG.apply(weights_init)
if config['netG'] != '':
    netG.load_state_dict(torch.load(config.netG))
print('Generator:\n\n', netG)

# discriminator
netD = Discriminator(num_gpu, ndf, nc).to(device)
netD.apply(weights_init)
if config['netD'] != '':
    netD.load_state_dict(torch.load(config.netD))
print('Discriminator:\n\n', netD)

# loss
criterion = nn.BCELoss()

# noise
fixed_noise = torch.randn(config['batch_size'], nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

# check single train iteration
if config['dry_run']:
    config['epochs'] = 1

# start training
print('TRAINING STARTED...')
for epoch in range(config['epochs']):
    for i, data in enumerate(data_loader, 0):

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        # train with real
        netD.zero_grad()
        real = data[0].to(device)
        batch_size = real.size(0)
        label = torch.full((batch_size,), real_label, dtype=real.dtype, device=device)

        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))

        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, config['epochs'], i, len(data_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            torchvision.utils.save_image(real,
                                         '%s/real_samples.png' % config['assets'],
                                         normalize=True)
            fake = netG(fixed_noise)
            torchvision.utils.save_image(fake.detach(),
                                         '%s/fake_samples_epoch_%03d.png' % (config['assets'], epoch),
                                         normalize=True)

        if config['dry_run']:
            break

    # do checkpointing for each epoch
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (config['assets'], epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (config['assets'], epoch))

print('TRAINING FINISHED...')
