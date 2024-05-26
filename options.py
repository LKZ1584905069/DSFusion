import argparse

class TrainOptions():
    def __init__(self):
         self.parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
         self.parser.add_argument('--state', default='train', help='模型状态 train or eval')
         self.parser.add_argument('--gpu', default=0, help='gup')
         self.parser.add_argument('--traindata', default='./dataset', help='训练数据存放路径')
         self.parser.add_argument('--testdata', default='./testimgs', help='测试数据存放路径')
         self.parser.add_argument('--batchsize', default=16, help='Batch大小')
         self.parser.add_argument('--epoch', default=20, help='训练轮数')
         self.parser.add_argument('--lr', default=1e-4, help='学习率')
         self.parser.add_argument('--pacturesize', default=128, help='图片大小')
         self.parser.add_argument('--saveModelPath', default='./model', help='第一阶段模型保存文件夹')
         self.parser.add_argument('--saveModelPath2', default='./model2', help='第二阶段模型保存文件夹')
         self.parser.add_argument('--saveDisPath', default='./dis', help='Dis保存文件夹')
         self.parser.add_argument('--savepicture', default='./result/', help='测试图片保存地址')
         self.parser.add_argument('--n_workers', default=1, help='dataloder的线程数')
         self.parser.add_argument('--Conv_out', default=40, help='encoder卷积的输出')
         self.parser.add_argument('--Loss_weight', default=[10,1,10,10], help='三个依次是grad、ssim、intenty、GAN')

    def parse(self):
          opt = self.parser.parse_args(args=[])
          return opt

