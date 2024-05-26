import numpy as np
import torch
from math import exp
import torch.nn.functional as F
# import kornia.filters as KF
from options import TrainOptions

opt = TrainOptions().parse()
# 方差计算 std_ir = std(img_ir)
def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    # 为求源图像方差铺垫, u
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    # mu_sq = mu.pow(2)
    # 求图像的方差，只需做两次卷积，一次是对原图卷积，一次是对原图的平方卷积，然后用后者减去前者的平方即可。
    # sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    # return sigma1
    return mu


# 创建高斯卷积权重，用于求SSIM时候的卷积核
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window


# 计算 ssim 损失函数
def mssim(img1, img2, window_size=11):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2


    (_, channel, height, width) = img1.size()

    # 创建卷积核，采用高斯卷积
    window = create_window(window_size, channel=channel).to(img1.device)

    # 对图像本身做卷积，用于后面求图像的方差核协方差
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    # 用于后面求图像的方差核协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 求图像的方差，只需做两次卷积，一次是对原图卷积，一次是对原图的平方卷积，然后用后者减去前者的平方即可。
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # 求两图的协方差，只需做三次卷积，第一次是对两图的乘积卷积，第二次和第三次分别对两图本身卷积，然后用
    # 第一次的卷积结果减去第二、三次卷积结果的乘积。
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # C1、C2两个常数，使分母不为 0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # SSIM的 分子之一, 协方差分子
    v1 = 2.0 * sigma12 + C2

    # SSIM的分母之一
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret





def final_ssim(img_ir, img_vis, img_fuse):

    # 获得两个源图像关于融合图的SSIM
    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    # 获得两个图像的方差
    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    # m = torch.mean(img_ir)
    # w_ir = torch.where(img_ir > m, one, zero)

    # torch.where(condition, x, y)
    # condition是条件，x 和 y 是同shape 的矩阵, 针对矩阵中的某个位置的元素, 满足条件就返回x，不满足就返回y
    # map1 和 map2 的数值刚好相反
    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    # ssim = ssim * w_ir
    return ssim.mean()

# 梯度算子
def grad(img):
    import numpy as np
    with torch.no_grad():
        kernel = np.array([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])
        kernel = np.expand_dims(kernel, axis=0)
        kernel = np.expand_dims(kernel, axis=0)
        kernel = torch.from_numpy(kernel).float()
        grad_img = torch.nn.Conv2d(1,1,3,stride=1,padding=1,dilation=1, groups=1,bias=False)
        grad_img.weight.data = kernel
        # grad_img.bais = torch.tensor(0).cuda()
        grad_img = grad_img(img)
        return grad_img


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=1):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        mask = torch.logical_and(img1>0,img2>0).float()

        for i in range(self.window_size//2):
            mask = (F.conv2d(mask, window, padding=self.window_size//2, groups=channel)>0.8).float()

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask=mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=1):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    #print(mask.shape,ssim_map.shape)
    ssim_map = ssim_map*mask

    ssim_map = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def intLoss(img):
    down = torch.nn.AvgPool2d(8)(img)
    one = torch.ones_like(down)
    zero = torch.zeros_like(down)
    img1 = torch.where(down > 0.5,one,zero)
    img2 = torch.nn.UpsamplingBilinear2d(scale_factor=8)(img1)
    return img2
