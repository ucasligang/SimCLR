import os

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# class CIFAR10Pair(CIFAR10):
#     """CIFAR10 Dataset.
#     """
#
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             pos_1 = self.transform(img)
#             pos_2 = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return pos_1, pos_2, target


class RadarPair(Dataset):
    """Radar Dataset.
    """
    def __init__(self,ROOT_PATH, setname, transform=None):
        csv_path = os.path.join(ROOT_PATH, setname, setname+'.csv')
        lines = [x.strip() for x in open(csv_path,'r').readlines()][1:]
        data = []
        self.data = data
        self.transform = transform
        for l in lines:
            _, name = l.split(',')
            path = os.path.join(ROOT_PATH, setname, 'data', name)
            data.append(path)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        # img = Image.fromarray(img)
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        return pos_1, pos_2

    def __len__(self):
        return len(self.data)


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

if __name__ == '__main__':
    data = RadarPair(ROOT_PATH='data_radar/', setname='train', transform=train_transform)
    pos1, pos2 = data.__getitem__(1)
    print(pos1)