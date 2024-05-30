from datetime import datetime
import torch
from options import TrainOptions
from torch.utils.data import DataLoader
from dataset import *
from model import Model
import os
from torch.optim import Adam,RMSprop
from loss import intLoss
import kornia.filters as KF
import torch.nn.functional as F
from test import test


# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 权重初始化
def gaussian_weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            print('卷积权重初始化失败')
    elif isinstance(m, torch.nn.BatchNorm2d):
        try:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        except:
            print('BN初始化失败')

def dataLoader(opt, dataset):
    return DataLoader(
            dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=opt.n_workers,
            drop_last=False,
         )


def train(opt, dataset):
    # 加载数据
    dataloader = dataLoader(opt, dataset)
    # 训练轮数
    train_num = len(dataset)
    # model
    print('\n--- load model ---')
    print(f'------ 训练轮数为{train_num} --------')
    model = Model()
    # 初始化权重，放在cuda前面
    model.apply(gaussian_weights_init)
    model.cuda()

    # 开始计时
    from datetime import datetime
    start_time = datetime.now()
    # count 用于打印数据，count%10==0打印损失函数，count%50=0保存模型一次
    count = 0
    batch_num = len(dataloader)
    # 训练的轮数， opt.epoch = 1
    for ep in range(opt.epoch):
        print('~~~Main_GAN 训练开始！~~~~')

        # 模型设置为train模式
        model.train()

        # 每个batchsize的训练
        # for it, (img_ir, img_vi, img_label) in enumerate(dataloader):
        for it, (img_ir, img_vi) in enumerate(dataloader):
            count += 1
            print(f'--第{ep}轮---{count} / {batch_num}----  ')

            # 设置网络优化器
            optimizer_G = Adam(model.parameters(), opt.lr)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.1)

            # 优化器梯度清零
            optimizer_G.zero_grad()


            # 图片放入cuda
            if opt.gpu:
                img_vi = img_vi.cuda()
                img_ir = img_ir.cuda()

            # 生成的图片名命为 gen_image
            gen_iamge = model(vis=img_vi, ir=img_ir)


           # 梯度损失
            # 使用Sobel函数求导
            grad_ir = KF.spatial_gradient(img_ir, order=2).abs().sum(dim=[1, 2])
            grad_vi = KF.spatial_gradient(img_vi, order=2).abs().sum(dim=[1, 2])
            grad_fus = KF.spatial_gradient(gen_iamge, order=2).abs().sum(dim=[1, 2])
            grad_joint = torch.max(grad_ir, grad_vi)
            # 第二步：求 vis 和 ir 中用不上的梯度
            zeros = torch.zeros_like(grad_vi)
            ones = torch.ones_like(grad_vi)
            vis_dis = torch.where(grad_vi - grad_ir > 0, ones, zeros)
            # ir_dis = torch.where(grad_vi-grad_ir <= 0,ones,zeros)
            ir_dis = 1 - vis_dis
            dis_vi = grad_vi * vis_dis  # [b,c,h,w]
            dis_ir = grad_ir * ir_dis  # [b,c,h,w]

            # 第三步：正相关是IF靠近联合梯度，负相关是IF原理用不上的梯度
            d_ap = torch.mean((grad_fus - grad_joint) ** 2)
            d_an_ir = torch.mean((grad_fus - dis_ir) ** 2)
            d_an_vi = torch.mean((grad_fus - dis_vi) ** 2)

            # 第四步：计算ContrastLoss
            loss_grad = d_ap / (d_an_vi + 1e-7) + d_ap / (d_an_ir + 1e-7)

            # loss_grad = F.l1_loss(grad_fus,grad_vi) + 5 * F.l1_loss(grad_fus,grad_ir)

            # SSIM整体损失
            # ssim_loss = SSIMLoss(window_size=11)
            # loss_ssim = 0.5 * ssim_loss(img_ir, gen_iamge) + 0.5 * ssim_loss(img_vi, gen_iamge)

            # SSIM局部损失
            # loss_ssim = final_ssim(img_ir, img_vi, gen_iamge)

            # 自己的MySSIM，这个的亮度和方差是分开的，方差用两个源图像，亮度用max(VIS,IR)
            # max_image = torch.maximum(img_ir,img_vi)
            # myssim = MySSIM(window_size=7)
            # ssim_loss = SSIMLoss(window_size=11)
            # loss_ssim = ssim_loss(img_ir,gen_iamge,model='D') + ssim_loss(img_vi,gen_iamge,model='L') +
            #              ssim_loss(max_image,gen_iamge,model='D')
            # loss_ssim = myssim(img_ir,gen_iamge,model='D') + myssim(img_vi,gen_iamge,model='D') + 2 * myssim(img_vi,gen_iamge,model='L')
            # loss_ssim = ssim_loss(max_image, gen_iamge, model='L')

            # 强度损失，做的是 img 和 label 的
            # loss_intensity = F.l1_loss(gen_iamge,img_ir) + F.l1_loss(gen_iamge, img_vi)

            # mean_int = img_vi / 2 + img_ir / 2
            # loss_intensity = F.l1_loss(gen_iamge,mean_int)

            # 强度损失：max( VIS, IR )
            # loss_intensity = F.l1_loss(gen_iamge,torch.maximum(img_ir,img_vi))

            # 强度损失，自己设计的
            choose_ir = intLoss(img_ir)
            # 如果choose_ir==1那么这块是高亮的热目标信息，需要保留；否则就是用vis的
            block_ir = torch.where(choose_ir > 0, ones, zeros)
            # block_vis = 1 - block_ir
            loss_intensity = F.l1_loss(gen_iamge*block_ir,img_ir*block_ir) + F.l1_loss(gen_iamge, img_vi)



            # loss_total = 2 * loss_grad + 5 * loss_ssim + 5 * loss_intensity
            # loss_total = loss_grad + 10 * loss_ssim
            loss_total = loss_grad + 5 * loss_intensity

            loss_total.backward()

            optimizer_G.step()

            scheduler.step()
# ----------------------------------------------------------------------------------------------------------

            # 打印损失函数
            if count % 50 == 0:
                elapsed_time = datetime.now() - start_time
                # print('loss_grad: %s,  loss_intensity: %s ,loss_content: %s, loss_d1:%s, loss_d3:%s, loss_total: %s,selapsed_time: %s' % (
                #     loss_grad.item(), loss_intensity.item(), loss_content.item(),d1_loss.item(),d3_loss.item(),loss_total.item(), elapsed_time))
                # print('loss_grad: %s,  loss_intensity: %s ,loss_ssim: %s, loss_total: %s,selapsed_time: %s' % (
                #     loss_grad.item(), loss_intensity.item(), loss_ssim.item(),loss_total.item(), elapsed_time))
                print('loss_grad: %s,  loss_ssim: %s, loss_total: %s,selapsed_time: %s' % (
                    loss_grad.item(),  loss_intensity.item(),loss_total.item(), elapsed_time))


            if count % 50 == 0:
                # save model
                model.eval()
                model.cpu()
                save_model_filename = "Epoch_" + str(count) + "_iters_" + str(count) + ".model"
                save_model_path = os.path.join(opt.saveModelPath ,save_model_filename)
                torch.save(model.state_dict(), save_model_path)
                model.train()
                model.cuda()

            #  每个300轮验证一次结果
            if count % 100 == 0:
                # 因为计算机是并行的cpu的写入不能满足代码的调用速度，所以使用本轮前面的保存的一个model
                modelPath = "Epoch_" + str(count-50) + "_iters_" + str(count-50) + ".model"
                # modelPath = "Epoch_" + str(250) + "_iters_" + str(250) + ".model"
                test(modelPath)

    # 训练结束，保存最后一个模型
    model.eval()
    model.cpu()
    save_model_filename = "Final_epoch_" + str(count) + ".model"
    # args.save_model_dir = models
    save_model_path = os.path.join(opt.saveModelPath, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)




if __name__ == '__main__':
    opt = TrainOptions().parse()

    is_train = True

    if is_train:
        # 数据加载
        dataset = dataset(opt.traindata)
        train(opt,dataset)

