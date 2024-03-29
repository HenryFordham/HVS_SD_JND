from __future__ import print_function
import torch.nn.functional as F
import cv2
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import models.GMSD as GMSD
from models.FSIM import FSIM
from models.GMSD import GMSD
from models.MS_SSIM import MS_SSIM
from models.SSIM import SSIM
from models.DISTS import DISTS
import real_resnet_predict_new

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return device

# 根据 Gradient 来生成 CAM 图
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

# 根据 Gradient 来生成 Feature CAM 图
def generateFeatureCAM(feature, gradient, image_shape):
    weights = F.adaptive_avg_pool2d(gradient, 1)
    GCAM = torch.mul(feature, weights)
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

# 求出两张图片的 PSNR
def psnr(original, contrast):
    mse = torch.mean((contrast - original).pow(2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

# 求出两张图片的 MSE
def mse(original, contrast):
    return torch.mean((contrast - original).pow(2))

# 根据图片地址来加载图片
def load_images(image_paths):
    images = []
    raw_images = []
    image, raw_image = preprocess(image_paths)
    images.append(image)
    print("Images Path is: " + image_paths)
    raw_images.append(raw_image)
    return images, raw_images

# 图片的预处理过程
def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    raw_image = cv2.imread(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image), raw_image


# 根据图片地址来加载图片
def load_images_PC(image_paths):
    images = []
    raw_images = []
    image, raw_image = preprocess_PC(image_paths)
    images.append(image)
    print("Images Path is: " + image_paths)
    raw_images.append(raw_image)
    return images, raw_images

# 图片的预处理过程
def preprocess_PC(image_path):
    image = Image.open(image_path)
    raw_image = cv2.imread(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image), raw_image

# 保存 CAMP 图像
# save_CMAP(filename = str("/root/IQA_result/_CAMP.png"),
#           gcam = [512, 512])
def save_CMAP(filename, gcam):
    gcam = gcam.detach().cpu().numpy()
    _range = np.max(gcam) - np.min(gcam)
    gcam = (gcam - np.min(gcam)) / _range

    gcam_ori = gcam
    camp_ori = cm.jet_r(gcam_ori)[..., :3] * 255.0
    cv2.imwrite(filename, np.uint8(camp_ori))
    return camp_ori

def save_Image(filename, image):
    gcam = image.detach().cpu().numpy()
    _range = np.max(gcam) - np.min(gcam)
    gcam = (gcam - np.min(gcam)) / _range

    gcam_ori = gcam
    camp_ori = cm.jet_r(gcam_ori)[..., :1] * 255.0
    cv2.imwrite(filename, np.uint8(camp_ori))
    return camp_ori

# 把 JND 图像处理后输出
def generateJND(jnd):
    B, C, H, W = jnd.shape
    jnd = jnd.view(B, -1)
    jnd -= jnd.min(dim=1, keepdim=True)[0]
    jnd /= jnd.max(dim=1, keepdim=True)[0]
    jnd = jnd.view(B, C, H, W)
    return jnd

# 保存 JND 图像单通道图像
# save_JND(filename = str("/root/IQA_result/_JND.png"),
#           jnd = [512, 512])
def save_JND_Single(filename, jnd):
    jnd = jnd.detach().cpu().numpy()
    _range = np.max(jnd) - np.min(jnd)
    jnd = (jnd - np.min(jnd)) / _range
    jnd_ori = jnd
    jnd_ori = cm.jet_r(jnd_ori)[..., :3] * 255.0
    cv2.imwrite(filename, np.uint8(jnd_ori))
    return jnd_ori

# 保存 Guided 图像
# save_Guided(
#        filename=str("/root/IQA_result/_Guided.png"),
#        gradient=grad.squeeze(0),
#    )
def save_Guided(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))

# 保存 Guided 单通道图像
# save_Guided_Single(filename=str("/root/IQA_result/_Guided_G.png"),
#             gcam=grad[0, 1])
def save_Guided_Single(filename, guided):
    guided = guided.detach().cpu().numpy()
    guided_ori = guided
    guided_ori = cm.jet_r(guided_ori)[..., :1]*255
    cv2.imwrite(filename, np.uint8(guided_ori))
    return guided_ori

def AIC_Magnitute(batch_size, original, distorted, magnitute, jnd, weight):
    device = get_device("cuda")
    pred_list = []
    pr = real_resnet_predict_new.pred()
    # IQA初始化：
    ssim_loss = SSIM(channels=3).to(device)
    fsim_loss = FSIM(channels=3).to(device)
    gmsd_loss = GMSD().to(device)
    ms_ssim_loss = MS_SSIM(channels=3).to(device)
    dists_loss = DISTS().to(device)
    total_IQA_loss = 0
    # 处理所有 batch 原图对应的 Grad-CAM 图像和剪裁后的图像
    for i in range(0, batch_size):
        with torch.no_grad():
            distorted_ty = pr.predi(distorted[i].unsqueeze(0))
            pred_list.append(distorted_ty)

    # 计算 AIC 损失
    # total_TV = torch.tensor(0.)
    total_Loss = torch.tensor(0.)
    # 求出一个 batch 的 magnitude
    kx = [[-1 / 3, 0, 1 / 3], [-1 / 3, 0, 1 / 3], [-1 / 3, 0, 1 / 3]]
    kx = torch.FloatTensor(kx).unsqueeze(0).unsqueeze(0).to(device)
    ky = [[-1 / 3, -1 / 3, -1 / 3], [0, 0, 0], [1 / 3, 1 / 3, 1 / 3]]
    ky = torch.FloatTensor(ky).unsqueeze(0).unsqueeze(0).to(device)
    for i in range(0, batch_size):
        i_num = 0
        total_TV = torch.tensor(0.)
        imgd = F.pad(input=magnitute[i].unsqueeze(0), pad=(1, 1, 1, 1), mode='replicate')
        imgd_list = imgd.split(1, dim=1)
        #print(len(imgd_list))
        #weights = F.adaptive_avg_pool2d(feature[i].unsqueeze(0).grad, 1)
        #print(weights.shape)
        #w = weight[i].unsqueeze(0)
        for single_imgd in imgd_list:
            # print("wshape", w[0, i, 0, 0], i_num)
            gradx = F.conv2d(single_imgd.to(device),
                             nn.Parameter(data=kx, requires_grad=False))
            grady = F.conv2d(single_imgd.to(device),
                             nn.Parameter(data=ky, requires_grad=False))
            TV = torch.sum(torch.sqrt((torch.pow(gradx, 2) + torch.pow(grady, 2))))
            total_TV = TV + total_TV
            i_num = i_num + 1
        # mean_TV = total_TV / batch_size
        T_1 = 1 / (total_TV + 1)
        T_2 = 1 /(torch.sum(jnd[i]) + 1)
        L_mask = torch.log((torch.pow(T_1, 2) + torch.pow(T_2, 2) + 0.001) / (2 * torch.mul(T_1, T_2) + 0.001))

        # print("LOSS TYPE", pred_list[i])
        if pred_list[i] == 0:
            L_1_0 = gmsd_loss(original[i].unsqueeze(0), distorted[i].unsqueeze(0),
                              as_loss=True)
            L_1_1 = ms_ssim_loss(distorted[i].unsqueeze(0), original[i].unsqueeze(0),
                                 as_loss=True)
            L_1_2 = dists_loss(original[i].unsqueeze(0), distorted[i].unsqueeze(0),
                               as_loss=True)
            L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)

        elif pred_list[i] == 1:
            L_1_0 = dists_loss(original[i].unsqueeze(0), distorted[i].unsqueeze(0),
                               as_loss=True)
            L_1_1 = ssim_loss(distorted[i].unsqueeze(0), original[i].unsqueeze(0),
                              as_loss=True)
            L_1_2 = gmsd_loss(original[i].unsqueeze(0), distorted[i].unsqueeze(0),
                              as_loss=True)
            L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)

        elif pred_list[i] == 2:
            L_1_0 = ms_ssim_loss(distorted[i].unsqueeze(0), original[i].unsqueeze(0),
                                 as_loss=True)
            L_1_1 = gmsd_loss(original[i].unsqueeze(0), distorted[i].unsqueeze(0),
                              as_loss=True)
            L_1_2 = fsim_loss(distorted[i].unsqueeze(0), original[i].unsqueeze(0),
                              as_loss=True)
            L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)
        elif pred_list[i] == 3:
            L_1_0 = ms_ssim_loss(distorted[i].unsqueeze(0), original[i].unsqueeze(0),
                                 as_loss=True)
            L_1_1 = gmsd_loss(original[i].unsqueeze(0), distorted[i].unsqueeze(0),
                              as_loss=True)
            L_1_2 = dists_loss(original[i].unsqueeze(0), distorted[i].unsqueeze(0),
                               as_loss=True)
            L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)
        else:
            L_1_0 = gmsd_loss(original[i].unsqueeze(0), distorted[i].unsqueeze(0),
                              as_loss=True)
            L_1_1 = ssim_loss(distorted[i].unsqueeze(0), original[i].unsqueeze(0),
                              as_loss=True)
            L_1_2 = ms_ssim_loss(distorted[i].unsqueeze(0), original[i].unsqueeze(0),
                                 as_loss=True)
            L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)

        total_IQA_loss = total_IQA_loss + L_1
        #print("total_IQA_loss Loss: ", total_IQA_loss)
        #print("L_1 Loss: ", L_1)
        loss = L_mask + L_1
        total_Loss = total_Loss + loss
    L_mask = L_mask / batch_size
    loss = total_Loss / batch_size
    IQA_average_batch_size_loss = total_IQA_loss / batch_size
    #return L_mask, loss, IQA_average_batch_size_loss
    return L_mask, IQA_average_batch_size_loss, loss

def AIC(predicted_type, original, distorted):
    device = get_device("cuda")

    # IQA初始化：
    ssim_loss = SSIM(channels=3).to(device)
    fsim_loss = FSIM(channels=3).to(device)
    gmsd_loss = GMSD().to(device)
    ms_ssim_loss = MS_SSIM(channels=3).to(device)
    dists_loss = DISTS().to(device)

    if predicted_type == 0:
        L_1_0 = gmsd_loss(original.unsqueeze(0), distorted.unsqueeze(0),
                          as_loss=True)
        L_1_1 = ms_ssim_loss(distorted.unsqueeze(0), original.unsqueeze(0),
                             as_loss=True)
        L_1_2 = dists_loss(original.unsqueeze(0), distorted.unsqueeze(0),
                           as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)

    elif predicted_type == 1:
        L_1_0 = dists_loss(original.unsqueeze(0), distorted.unsqueeze(0),
                           as_loss=True)
        L_1_1 = ssim_loss(distorted.unsqueeze(0), original.unsqueeze(0),
                          as_loss=True)
        L_1_2 = gmsd_loss(original.unsqueeze(0), distorted.unsqueeze(0),
                          as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)

    elif predicted_type == 2:
        L_1_0 = ms_ssim_loss(distorted.unsqueeze(0), original.unsqueeze(0),
                             as_loss=True)
        L_1_1 = gmsd_loss(original.unsqueeze(0), distorted.unsqueeze(0),
                          as_loss=True)
        L_1_2 = fsim_loss(distorted.unsqueeze(0), original.unsqueeze(0),
                          as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)
    elif predicted_type == 3:
        L_1_0 = ms_ssim_loss(distorted.unsqueeze(0), original.unsqueeze(0),
                             as_loss=True)
        L_1_1 = gmsd_loss(original.unsqueeze(0), distorted.unsqueeze(0),
                          as_loss=True)
        L_1_2 = dists_loss(original.unsqueeze(0), distorted.unsqueeze(0),
                           as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)
    else:
        L_1_0 = gmsd_loss(original.unsqueeze(0), distorted.unsqueeze(0),
                          as_loss=True)
        L_1_1 = ssim_loss(distorted.unsqueeze(0), original.unsqueeze(0),
                          as_loss=True)
        L_1_2 = ms_ssim_loss(distorted.unsqueeze(0), original.unsqueeze(0),
                             as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)
    return L_1

def AIC_mnd(predicted_type, original, distorted):
    device = get_device("cuda")

    # IQA初始化：
    ssim_loss = SSIM(channels=3).to(device)
    fsim_loss = FSIM(channels=3).to(device)
    gmsd_loss = GMSD().to(device)
    ms_ssim_loss = MS_SSIM(channels=3).to(device)
    dists_loss = DISTS().to(device)

    if predicted_type == 0:
        L_1_0 = gmsd_loss(original, distorted, as_loss=True)
        L_1_1 = ms_ssim_loss(original, distorted, as_loss=True)
        L_1_2 = dists_loss(original, distorted, as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)

    elif predicted_type == 1:
        L_1_0 = dists_loss(original, distorted, as_loss=True)
        L_1_1 = ssim_loss(original, distorted, as_loss=True)
        L_1_2 = gmsd_loss(original, distorted, as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)

    elif predicted_type == 2:
        L_1_0 = ms_ssim_loss(original, distorted, as_loss=True)
        L_1_1 = gmsd_loss(original, distorted, as_loss=True)
        L_1_2 = fsim_loss(original, distorted, as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)
    elif predicted_type == 3:
        L_1_0 = ms_ssim_loss(original, distorted, as_loss=True)
        L_1_1 = gmsd_loss(original, distorted, as_loss=True)
        L_1_2 = dists_loss(original, distorted, as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)
    else:
        L_1_0 = gmsd_loss(original, distorted, as_loss=True)
        L_1_1 = ssim_loss(original, distorted, as_loss=True)
        L_1_2 = ms_ssim_loss(original, distorted, as_loss=True)
        L_1 = ((1 / 3) * L_1_0) + ((1 / 3) * L_1_1) + ((1 / 3) * L_1_2)
    return L_1


def SSIM_Magnitute_(batch_size, original, distorted, magnitute, jnd, weight):
    total_IQA_loss = 0
    total_Loss = torch.tensor(0.)
    total_MASK = torch.tensor(0.)

    # 求出一个 batch 的 magnitude
    for i in range(0, batch_size):
        # imgd: torch.Size([128, 22, 22])
        # jndd: torch.Size([128, 22, 22])
        imgd = magnitute[i]
        jndd = jnd[i]
        _range = torch.max(weight) - torch.min(weight)
        weight = (weight - torch.min(weight)) / _range
        N = F.adaptive_avg_pool2d(imgd, 1)
        N0 = 0.1 * F.adaptive_avg_pool2d(jndd, 1)
        L_mask = torch.log((torch.pow(torch.sum(N), 2) + torch.pow(torch.sum(N0), 2) + 0.001) /
                           (2 * torch.mul(torch.sum(N), torch.sum(N0)) + 0.001))
        ms_ssim_loss = MS_SSIM(channels=3).to("cuda")
        L_1 = ms_ssim_loss(distorted, original,
                     as_loss=True)
        total_MASK = total_MASK + L_mask
        total_IQA_loss = total_IQA_loss + L_1
        loss = 0.1 * L_mask + L_1
        total_Loss = total_Loss + loss
    total_MASK /= batch_size
    total_IQA_loss /= batch_size
    total_Loss /= batch_size
    return total_MASK, total_IQA_loss, total_Loss


def AIC_Magnitute_(batch_size, original, distorted, magnitute, jnd, weight):
    device = get_device("cuda")
    pred_list = []
    pr = real_resnet_predict_new.pred()
    total_IQA_loss = 0
    # 处理所有 batch 原图对应的 Grad-CAM 图像和剪裁后的图像
    for i in range(0, batch_size):
        with torch.no_grad():
            distorted_ty = pr.predi(distorted[i].unsqueeze(0))
            pred_list.append(distorted_ty)

    # 计算 AIC 损失
    total_TV = torch.tensor(0.)
    total_Loss = torch.tensor(0.)
    total_MASK = torch.tensor(0.)

    # 求出一个 batch 的 magnitude
    for i in range(0, batch_size):
        # imgd: torch.Size([128, 22, 22])
        # jndd: torch.Size([128, 22, 22])
        imgd = magnitute[i]
        jndd = jnd[i]
        _range = torch.max(weight) - torch.min(weight)
        weight = (weight - torch.min(weight)) / _range
        # N : torch.Size([128, 1, 1])
        # N0 : torch.Size([128, 1, 1])
        #N = torch.mul(weight, F.adaptive_avg_pool2d(imgd, 1))
        norm = torch.nn.BatchNorm2d(128,
                         eps=1e-05,
                         momentum=0.1,
                         affine=True,
                         track_running_stats=True).to("cuda")
        # print(F.adaptive_avg_pool2d(imgd, 1).shape)
        # weight = norm(weight.to("cuda"))
        #imgd = norm(imgd.to("cuda").unsqueeze(0)).squeeze(0)
        #jndd = norm(jndd.to("cuda").unsqueeze(0)).squeeze(0)
        
        #N = torch.mul(weight[i], F.adaptive_avg_pool2d(imgd, 1))
        N = F.adaptive_avg_pool2d(imgd, 1)
        #print(N)
        #N = torch.mul(weight[i], N)
        #_range = torch.max(jndd) - torch.min(jndd)
        #jndd = (jndd - torch.min(jndd)) / _range
        #N0 = 0.1* torch.mul(weight[i],  F.adaptive_avg_pool2d(jndd, 1)) 
        #N0 = torch.mul(100000000*weight[i],  F.adaptive_avg_pool2d(jndd, 1))


        
        N0 =  0.1 * F.adaptive_avg_pool2d(jndd, 1)
        # N0 = F.adaptive_avg_pool2d(jndd, 1)
        # N0 = torch.mul(weight[i], N0)
        # L_mask tensor(4.1004, device='cuda:0', grad_fn=<LogBackward>)
        #L_mask = N0
        L_mask = torch.log((torch.pow(torch.sum(N), 2) + torch.pow(torch.sum(N0), 2) + 0.001) /
                           (2 * torch.mul(torch.sum(N), torch.sum(N0)) + 0.001))
        L_1 = AIC(pred_list[i], original[i], distorted[i])
        total_MASK = total_MASK + L_mask
        total_IQA_loss = total_IQA_loss + L_1
        loss = 0.1 * L_mask + L_1
        # loss = L_mask + L_1
        total_Loss = total_Loss + loss
    total_MASK /= batch_size
    total_IQA_loss /= batch_size
    total_Loss /= batch_size
    return total_MASK, total_IQA_loss, total_Loss

