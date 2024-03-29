import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
import math


class Dataset_Plain(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(Dataset_Plain, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'pair'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize*self.sf
        self.paths = util.get_image_paths(opt['dataroot'])

    def __getitem__(self, index):
        # ------------------------------------
        # get image
        # ------------------------------------
        path = self.paths[index]
        img = util.imread_uint(path, self.n_channels)

        # ------------------------------------
        # pre-processing
        # ------------------------------------
        H, W, _ = img.shape
        H = math.floor(H / 128) * 128
        W = math.floor(W / 128) * 128

        patch_L = img[0:0 + H, 0:0 + W, :]
        img_L = util.uint2tensor3(patch_L)
        return {'img': img_L, 'path': path}

    def __len__(self):
        return len(self.paths)
