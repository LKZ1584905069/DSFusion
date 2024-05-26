import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF

class dataset(Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """
    # TODO: remove ground truth reference
    def __init__(self, path):
        super(dataset, self).__init__()
        self.vis_folder = os.path.join(path, 'vi')
        self.ir_folder = os.path.join(path, 'ir')
        # self.label_floder = os.path.join(path,'label')

        # gain infrared and visible images list
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        # self.label_list = sorted(os.listdir(self.label_floder))

        # print(len(self.vis_list), len(self.ir_list),len(self.label_list))
        print(len(self.vis_list), len(self.ir_list))
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        # gain image path
        ir_image_name = self.ir_list[index]
        vis_image_name = self.vis_list[index]
        # label_image_name = self.label_list[index]

        vis_path = os.path.join(self.vis_folder, vis_image_name)
        ir_path = os.path.join(self.ir_folder, ir_image_name)
        # label_path = os.path.join(self.label_floder,label_image_name)


        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        # label = self.imread(path=label_path)

        # return ir, vis, label
        return ir, vis

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, label=False):
        img = Image.open(path)
        im_ts = TF.to_tensor(img)
        return im_ts