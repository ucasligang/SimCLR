import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = 'materials/'


class ImageDataset(Dataset):

    def __init__(self, ROOT_PATH, setname):
        csv_path = osp.join(ROOT_PATH, setname+'.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.474, 0.461, 0.420],
                                 std=[0.240, 0.237, 0.240])
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        data = self.transform(Image.open(path).convert('RGB'))
        return data, label

