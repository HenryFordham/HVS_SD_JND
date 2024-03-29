from __future__ import print_function
import logging
import sys, os
import torch.nn.functional as F
import grad_cam_usage as grad_cam
from models.AutoEncoder2 import Concatenation
import torch
import cv2

import argparse
from PIL import Image
from torchvision import transforms
import numpy as np

# 设置一系列学习的参数，学习率，batch_size 和 图像大小
gpu_num = torch.cuda.device_count()
ct = 0.000
print_freq = 10
base_lr = 1e-4
batch_size = 4
image_size = 176
global_step = 0
tot_epoch = 1000000
tot_step = 2500000
cur_lr = base_lr = 1e-4
lr_decay = 0.1
decay_interval = 2200000
tb_logger = None
logger = logging.getLogger("AutoEncoder")
warmup_step = 0

# 加载模型
def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

# 加载图片
def load_images(image_paths):
    images = []
    raw_images = []
    image, raw_image = preprocess(image_paths)
    images.append(image)
    print("Images Path is: " + image_paths)
    raw_images.append(raw_image)
    return images, raw_images

def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    raw_image = cv2.imread(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image), raw_image

# 计算图片的 PSNR
def psnr(original, contrast):
    mse = torch.mean((contrast - original).pow(2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return device

def get_type(device):
    if (device.type == "cpu"):
        torchType = torch.FloatTensor
    else:
        torchType = torch.cuda.FloatTensor
    return torchType

def generateCAM(feature, gradient, image_shape):
    weights = F.adaptive_avg_pool2d(gradient, 1)
    GCAM = torch.mul(feature, weights).sum(dim=1, keepdim=True)
    GCAM = torch.abs(GCAM)
    GCAM = F.relu(GCAM)
    GCAM = F.interpolate(
        GCAM, image_shape, mode="bilinear", align_corners=False
    )
    B, C, H, W = GCAM.shape
    GCAM = GCAM.view(B, -1)
    GCAM -= GCAM.min(dim=1, keepdim=True)[0]
    GCAM /= GCAM.max(dim=1, keepdim=True)[0]
    GCAM = GCAM.view(B, C, H, W)
    return GCAM

def stage1_proceed(images, cuda):
    # 加载编码器的网络
    from models.StageOneModel import ImageCompressor as ImC
    model = ImC()

    # 把预训练的模型给加载好
    load_model(model, "checkpoints/iter_20000.pth.tar")
    model.to(cuda)
    model.eval()

    # 使用编码器网络计算 Grad CAM
    gcam = grad_cam.GradCAM(model=model)

    # 将图像前向传播，这样能够获取到具体的特征层
    images = images.to(cuda)
    distortion_loss, clipped_recon_image, feature, F = gcam.forward(images)
    clipped_recon_image = clipped_recon_image.squeeze(0)

    # Grad-CAM
    image_grad = gcam.backward(distortion_loss=distortion_loss)
    GCAM = generateCAM(feature, feature.grad, gcam.image_shape)
    return clipped_recon_image, feature, GCAM, image_grad, F

def GetFeature(images):
    model = Concatenation()
    load_model(model, "checkpoints/Satisfy.pth.tar")
    device = get_device("cuda")
    net = model.to(device)
    device = get_device("cuda")
    if (device.type == "cpu"):
        torchType = torch.FloatTensor
    else:
        torchType = torch.cuda.FloatTensor
    clipped_recon_image, feature, GCAM, grad, F = stage1_proceed(images, device)

    # 输出梯度图像
    _range = torch.max(grad) - torch.min(grad)
    grad = (grad - torch.min(grad)) / _range
    concat = torch.cat((images.to(device), GCAM.to(device)), 1)
    concat = torch.cat((concat, torch.mul(GCAM, grad)), 1)

    jnd = net(concat)
    jnd = torch.nn.functional.interpolate(input=jnd, size=(176, 176), mode='bilinear',
                                          align_corners=False)
    jnd = torch.abs(jnd)
    jnd = torch.clamp(jnd, 0, 1)

    # 行列 176 * 176，是一张图片随即加入的噪声，非 1 即 -1
    col, row = images.shape[2], images.shape[3]
    mask = np.random.random((col, row))
    randnum = np.zeros((col, row))
    for i in range(col):
        for j in range(row):
            if mask[i][j] >= 0.5:
                randnum[i][j] = 1
            else:
                randnum[i][j] = -1
    # numpy 类型要转换成 torch 支持的类型，同时要是浮点类型的
    randnum = torch.from_numpy(randnum)
    randnum = randnum.to(device)
    batch_img2_ori = images.to(device) + 15*randnum * jnd.to(device)
    batch_img2_ori = torch.clamp(batch_img2_ori, 0, 1)
    batch_img2_ori.requires_grad_()
    clipped_recon_image2, feature2, GCAM2, image_grad2, F2 = stage1_proceed(batch_img2_ori, device)
    return batch_img2_ori

def HVS_SD_JND(picPath,args):
    # 加载图像
    device = get_device("cuda")
    torchType = get_type(device)
    images, raw_images = load_images(picPath)
    images = torch.stack(images).to(device)

    # 加载 stage one 的失真图像, lmbda 为 JND 的参数 lmbda
    batch_img2_ori = GetJNDdistorted(images, lmbda=args.lmbda)
    batch_img2_ori = batch_img2_ori.type(torchType)

    # 进入 stage two 的过程，但测试中用不到相关数据
    batch_img2_ori.requires_grad_()
    clipped_recon_image2, feature2, GCAM2, image_grad2, F2 = stage1_proceed(batch_img2_ori, device)

    # 输出 PSNR 信息
    psnr1 = psnr(images[0][0].unsqueeze(0).unsqueeze(0) * 255,
                 batch_img2_ori[0][0].unsqueeze(0).unsqueeze(0) * 255).item()
    psnr2 = psnr(images[0][1].unsqueeze(0).unsqueeze(0) * 255,
                 batch_img2_ori[0][1].unsqueeze(0).unsqueeze(0) * 255).item()
    psnr3 = psnr(images[0][2].unsqueeze(0).unsqueeze(0) * 255,
                 batch_img2_ori[0][2].unsqueeze(0).unsqueeze(0) * 255).item()
    total_psnr = psnr(images * 255, batch_img2_ori * 255).item()
    print("Result-R PSNR:", psnr1)
    print("Result-G PSNR:", psnr2)
    print("Result-B PSNR:", psnr3)
    print("Total PSNR: ", total_psnr)

def GetJNDdistorted(images2, lmbda):
    # 经过 stage one 的图像
    device = get_device("cuda")
    torchType = get_type(device)
    clipped_recon_image, feature, GCAM, grad, F = stage1_proceed(images2, device)
    _range = torch.max(grad) - torch.min(grad)
    grad = (grad - torch.min(grad)) / _range

    # 将梯度信息和 grad 和 grad-cam 融合
    concat = torch.cat((images2, GCAM), 1)
    concat = torch.cat((concat, torch.mul(GCAM, grad)), 1)

    # 加载 JND 网络
    model = Concatenation()
    load_model(model, "checkpoints/Satisfy.pth.tar")
    net = model.to(device)
    jnd = net(concat)
    jnd = torch.nn.functional.interpolate(input=jnd, size=(images2.shape[2], images2.shape[3]), mode='bilinear',
                                          align_corners=False)
    jnd = torch.abs(jnd)
    jnd = torch.clamp(jnd, 0, 1)

    # 一张图片随即加入的噪声，非 1 即 -1
    col, row = images2.shape[2], images2.shape[3]
    mask = np.random.random((col, row))
    randnum = np.zeros((col, row))
    for i in range(col):
        for j in range(row):
            if mask[i][j] >= 0.5:
                randnum[i][j] = 1
            else:
                randnum[i][j] = -1

    randnum = torch.from_numpy(randnum)
    randnum = randnum.type(torchType)
    batch_img2_ori = images2 + lmbda * randnum * jnd
    batch_img2_ori = torch.clamp(batch_img2_ori, 0, 1)
    return batch_img2_ori

def parse_args(argv):
    parser = argparse.ArgumentParser(description='HVS_SD_JND')
    parser.add_argument('--path', help='path of test pictures ', type=str)
    parser.add_argument('--lmbda', help='lmbda of JND ', type=int, default=1)
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)

    # 设置原图图像路径
    picPath = args.path
    file_paths = []

    # 取出所有图像
    for root, dirs, files in os.walk(picPath):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    for path in file_paths:
        HVS_SD_JND(path, args)

if __name__ == "__main__":
    main(sys.argv[1:])



