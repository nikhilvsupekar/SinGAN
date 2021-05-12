from __future__ import print_function
from scipy.ndimage.fourier import _get_output_fourier

from torch import embedding
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments

import cv2
import numpy as np
from matplotlib import pyplot as plt

def generate_gif(Gs,Zs,reals,NoiseAmp,opt,alpha=0.1,beta=0.9,start_scale=2,fps=10):

    in_s = torch.full(Zs[0].shape, 0, device=opt.device)
    images_cur = []
    count = 0

    for G,Z_opt,noise_amp,real in zip(Gs,Zs,NoiseAmp,reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        nzx = Z_opt.shape[2]
        nzy = Z_opt.shape[3]
        #pad_noise = 0
        #m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        if count == 0:
            z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
            z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
            z_prev1 = 0.95*Z_opt +0.05*z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95*Z_opt +0.05*functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
            z_prev2 = Z_opt

        for i in range(0,100,1):
            if count == 0:
                z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*z_rand
            else:
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*(functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device))

            z_curr = alpha*Z_opt+(1-alpha)*(z_prev1+diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                #I_prev = functions.upsampling(I_prev,reals[count].shape[2],reals[count].shape[3])
                I_prev = m_image(I_prev)
            if count < start_scale:
                z_curr = Z_opt

            z_in = noise_amp*z_curr+I_prev
            I_curr, embedding = G(z_in.detach(),I_prev)

            if (count == len(Gs)-1):
                I_curr = functions.denorm(I_curr).detach()
                I_curr = I_curr[0,:,:,:].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0)*255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        count += 1
    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs('%s/start_scale=%d' % (dir2save,start_scale) )
    except OSError:
        pass
    imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save,start_scale,alpha,beta),images_cur,fps=fps)
    del images_cur

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50):
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []

    # list of tuples (image, embedding)
    image_embeddings = []

    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
                #I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                # I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                I_prev = imresize_to_shape(I_prev, reals[n].permute(0, 2, 3, 1).shape[1:], opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

                # I_prev = m(I_prev)
                # I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                # I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr, embedding = G(z_in.detach(),I_prev)

            image_embeddings.append((reals[n], embedding))

            if n == len(reals)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
                    #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                    #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
            images_cur.append(I_curr)
        n+=1
    # return I_curr.detach()
    return image_embeddings




def edge_detector(image_path, sr_factor, t1=100, t2=200, aperture=3, blur_first=False, blur_kernel_size=(5,5)):
    """returns image with detected edges and a list of edge pixels"""
    img1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, dsize=(img1.shape[1] * sr_factor, img1.shape[0] * sr_factor))
    if blur_first:
        img1 = cv2.blur(img1, ksize=blur_kernel_size)
    img1 = cv2.Canny(img1, threshold1=t1, threshold2=t2, apertureSize=aperture)
    edgepixel_list = list(zip(np.nonzero(img1)[0], np.nonzero(img1)[1]))
    return img1, edgepixel_list


def proper_neighbors(p1, p2):
    return (np.linalg.norm(np.array(p1) - np.array(p2)) <= np.linalg.norm(np.array([1, 1]))) and (p1 != p2)


def interpolate_edge(e, sr_factor):
    p1, p2 = e 
    xs = list(np.linspace(p1[0], p2[0], sr_factor + 1, endpoint=True))
    ys = list(np.linspace(p1[1], p2[1], sr_factor + 1, endpoint=True))
    return list(zip(xs, ys))

def get_HR_edge_pixels(edge_px, sr_factor):
    edge_list = []

    for p1 in edge_px:
        for p2 in edge_px:
            if proper_neighbors(p1, p2):
                if p1[0] <= p2[0] and (p1, p2) not in edge_list:
                    edge_list.append((p1, p2))
                elif (p2, p1) not in edge_list:
                    edge_list.append((p2, p1))
    
    interpolated_edges = []
    for e in edge_list:
        interpolated_edges.append(interpolate_edge(e, sr_factor))
    
    hr_pixels = [p for e in interpolated_edges for p in e]

    return np.array(hr_pixels)

