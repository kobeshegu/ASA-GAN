import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from models import Generator


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--end_iter', type=int, default=10)

    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=8, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=5000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=256)
    parser.set_defaults(big=False)
    args = parser.parse_args()

    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)#, big=args.big )
    net_ig.to(device)
    for epoch in [10000*i for i in range(args.start_iter, args.end_iter+1)]:
        ckpt = './train_results/Panda256/models/%d.pth'%(epoch)
        checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
        net_ig.load_state_dict(checkpoint['g'])
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print('load checkpoint success, epoch %d'%epoch)

        net_ig.to(device)

        del checkpoint

        dist = './train_results/Panda256/pngeval_%d'%(epoch)
        dist = os.path.join(dist, 'img')
        os.makedirs(dist, exist_ok=True)

        with torch.no_grad():
            for i in tqdm(range(args.n_sample//args.batch)):
                noise = torch.FloatTensor(args.batch,noise_dim).normal_(0, 1).to(device)
               #  noise_z = torch.randn(args.batch, noise_dim)
               # noise_z = torch.FloatTensor(current_batch_size, nz).normal_(0, 1)
               #  noisetensor = torch.Tensor(args.batch, noise_dim)
               #  noise_zin = torch.nn.init.uniform_(tensor=noisetensor, a=-0.1, b=0.1)
               #  noise_zsig = torch.nn.init.constant_(tensor=noisetensor, val=0.90)
               #  noise_zuni = torch.ones(args.batch, noise_dim) - noise_zsig
               #  noise = torch.add(torch.mul(noise_zuni, noise_zin), torch.mul(noise_z, noise_zsig)).to(device)
               #no
                g_imgs = net_ig(noise)[0]
                g_imgs = F.interpolate(g_imgs, 256)
                for j, g_img in enumerate( g_imgs ):
                    vutils.save_image(g_img.add(1).mul(0.5), 
                        os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))
