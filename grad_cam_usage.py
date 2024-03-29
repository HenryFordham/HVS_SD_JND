#!/usr/bin/env python
# coding: utf-8

from collections import Sequence
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return device

train_lambda = 512
cur_lr = base_lr = 1e-4 # * gpu_num
class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    # 应该返回最终前向传播最后一层网络
    # 执行网络的 forward 函数
    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.image = image.requires_grad_()
        self.image.retain_grad()

        # 在次前向传播求出求出对应模型重的 mmsim 值
        clipped_recon_image, mse_loss, bpp, feature, F= self.model(image)

        # print("debug", clipped_recon_image.shape, " ", mse_loss, " ", bpp)
        # psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
        # print("psnr_ex", psnr)

        distribution_loss = bpp

        #self.rd_loss = train_lambda * distortion + distribution_loss
        self.rd_loss = mse_loss
        return self.rd_loss, clipped_recon_image, feature, F

    def backward(self, distortion_loss):
        """
        Class-specific backpropagation
        """
        self.distortion_loss = distortion_loss
        self.model.zero_grad()
        # print("mssim_loss is: " + str(self._ssim_loss))
        # self.logits.backward(retain_graph=True, gradient=self.rd_loss)
        self.distortion_loss.backward(retain_graph=True)
        # return self.image.grad
        return F.relu(self.image.grad)
        # 测试一下平均梯度大小
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is not None:
        #             print("{}, gradient: {}".format(name, param.grad.mean()))
        #         else:
        #             print("{} has not gradient".format(name))

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """

    def __init__(self, model, candidate_layers=None):
        # model 属于传入的模型，在编码器中 Analysis_net_17/Synthesis_net_17
        # 如果是自己设计的网络应该是 encoder 和 decoder
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        # candidate_layers 代表提取的是第几层的待选特征
        self.candidate_layers = candidate_layers

        # def save_fmaps(key):
        #     def forward_hook(module, input, output):
        #         # key 是每一层的名字，比如: Encoder.conv2 代表第二层卷积层
        #         if (key != ""):
        #             # print("key: " + str(key))
        #             # print("size: "+str(output.size()))
        #             self.fmap_pool[key] = output.detach()
        #     return forward_hook
        #
        # def save_grads(key):
        #     def backward_hook(module, grad_in, grad_out):
        #         # if (key != ""):
        #         # print("key: " + str(key))
        #         self.grad_pool[key] = grad_out[0].detach()
        #         # print("size: " + str(key)+str(grad_out[0].shape))
        #     return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        # for name, module in self.model.named_modules():
        #     if self.candidate_layers is None or name in self.candidate_layers:
        #         self.handlers.append(module.register_forward_hook(save_fmaps(name)))
        #         self.handlers.append(module.register_backward_hook(save_grads(name)))



    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = torch.abs(gcam)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )
        # print("shape is : ", str(gcam.shape))
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        # print("shape is : ", str(gcam.shape))
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam



class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient

class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))
