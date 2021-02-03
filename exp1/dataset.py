import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from glob import glob

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

testset = datasets.CIFAR10(
    root='/home/DATASET/CIFAR/', train=False, download=True, transform=test_transforms)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=4) #batch size set to 1

class QDataset(Dataset):
    def __init__(self,transform):
        self.imagelist=glob("./data/q/*.jpg")
        self.len=len(self.imagelist)
        self.transform=transform
    def __getitem__(self, index):
        image_path = self.imagelist[index]
        label=int(image_path.split("_")[0][-1])
        image = Image.open(image_path)
        return self.transform(image),label

    def __len__(self):
        return self.len