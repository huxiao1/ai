import torch 
import torch.nn as nn

import os

from PIL import Image
from torchvision import transforms

tran = transforms.Compose([
    #transforms.Resize((112, 112), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


class AlexNet2(nn.Module):
	def __init__(self,classes_num = 10):
		super(AlexNet2, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)		

		self.layer4 = nn.Sequential(
			nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)

		self.layer5 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.layer6 = nn.Sequential(
			nn.dropout(),
			nn.Linear(in_features=256*2*2, out_features=4096),
			nn.ReLU(inplace=True)
		)

		self.layer7 = nn.Sequential(
			nn.dropout(),
			nn.Linear(in_features=4096, out_features=4096),
			nn.ReLU(inplace=True)
		)

		self.layer8 = nn.Sequential(
			nn.Linear(in_features=4096, out_features=classes_num)
		)

	def forward(self,x):
		x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
		x = x.view(x.size(0), 256*2*2)
		x = self.layer8(self.layer7(self.layer6(x)))
		return x
